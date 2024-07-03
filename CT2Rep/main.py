import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from lightning.pytorch import LightningDataModule
from models.ct2rep import CT2RepModel
from modules.data_ct import CTReportDataset
import os
import torch.distributed as dist
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning_tools.callbacks import add_callbacks
from lightning.pytorch import seed_everything

def setup_distributed():
    """
    Initialize the distributed training environment.
    """
    # os.environ['MASTER_ADDR'] = 'localhost'
    # os.environ['MASTER_PORT'] = '12355'
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4"
    dist.init_process_group(backend='nccl')

import lightning.pytorch as pl

class CustomModel(pl.LightningModule):
    def __init__(self, model, criterion, optimizer, scheduler, args):
        super(CustomModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.args = args
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0
        self.best_recorder = {'val': {'bleu_4': self.val_score}}
        self.save_hyperparameters()

    def forward(self, images, reports_ids=None, reports_masks=None, mode='train'):
        return self.model(images, reports_ids, mask=reports_masks, mode=mode)

    def training_step(self, batch, batch_idx):
        images_id, images, reports_ids, reports_masks = batch
        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
        output = self(images, reports_ids, reports_masks, mode='train')
        loss = self.criterion(output, reports_ids, reports_masks)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def score(self, ref, hypo):
        return compute_scores(ref, hypo)

    def validation_step(self, batch, batch_idx):
        images_id, images, reports_ids, reports_masks = batch
        images, reports_ids, reports_masks = images.to(self.device), reports_ids.to(self.device), reports_masks.to(self.device)
        output = self(images, mode='sample')
        reports = self.model.tokenizer.decode_batch(output.cpu().numpy())
        ground_truths = self.model.tokenizer.decode_batch(reports_ids.cpu().numpy())
        val_met = self.score(ground_truths, reports)
        self.val_step_outputs.append({"hypo": reports, "ref": ground_truths, "id": images_id})
        return ground_truths, reports

    def on_validation_epoch_end(self):
        ref, hypo, ids = [], [], []
        for i in self.val_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}

        eval_res = self.metric_ftns(ref, hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.save_dir, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        weights = [1] * len(eval_res)

        for score_type, weight in zip(list(eval_res.keys()), weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
                self.best_recorder['val'].update(log)

        self._print_best()
        self.val_step_outputs.clear()

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format('bleu_4'))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.save_dir, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step_{}_bleu{:3f}.pth".format(current_epoch, global_step, eval_res['bleu_4']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)

    def configure_optimizers(self):
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()

def parse_agrs():

    # Example fromm teh repository
    # python main.py --max_seq_length 300 --threshold 10 --epochs 100 --save_dir results/test_ct2rep/ --step_size 1 --gamma 0.8 --batch_size 1 --d_vf 512
    parser = argparse.ArgumentParser()

    # Data loader settings
    parser.add_argument('--max_seq_length', type=int, default=300, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=10, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=16, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=1, help='the number of samples for a batch')
    parser.add_argument('--dataset_name', type=str, default='ct_dataset', help='dataset name.')
    parser.add_argument('--validate', default=True, type=bool, help="only run validation set")

    # Model settings (for Transformer)
    # parser.add_argument('--d_model', type=int, default=4096, help='the dimension of Transformer.')
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=512, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')

    # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=2, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='/network/scratch/b/bonaventure.dossou/probe_medical/models_v2/', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='/network/scratch/b/bonaventure.dossou/probe_medical/models_v2/records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=1, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.8, help='the gamma of the learning rate scheduler.')

    # # ====================== Pytorch Lightning ===========================
    # parser.add_argument('--devices', type=int, default=5, help='how many gpus to use')
    # parser.add_argument('--num_nodes', type=int, default=1, help='Number of GPU nodes for distributed training.')
    # parser.add_argument('--accelerator', type=str, default="gpu", choices=["cpu", "gpu", "tpu", "ipu", "hpu", "mps"], help='accelerator types')
    # parser.add_argument('--strategy', type=str, default="ddp", help='default ddp for multi-gpus')
    # parser.add_argument('--precision', type=str, default='bf16-mixed', help='16 or 32 bf16-mixed, using for original pytorch amp auto cast')
    # parser.add_argument('--limit_val_batches', type=float, default=1.0, help='How much of validation dataset to check (float = fraction, int = num_batches).')
    # parser.add_argument('--limit_test_batches', type=float, default=1.0, help='How much of test dataset to check (float = fraction, int = num_batches).')
    # parser.add_argument('--limit_train_batches', type=float, default=1.0, help='How much of training dataset to check (float = fraction, int = num_batches)')
    # parser.add_argument('--max_epochs', type=int, default=3, help='Stop training once this number of epochs is reached')
    # parser.add_argument('--every_n_train_steps', type=int, default=100, help='How many training steps to save a checkpoint')
    # parser.add_argument('--val_check_interval', type=float, default=0.2, help='How often to check the validation set') # default=1.0
    # parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Accumulates gradients over k batches before stepping the optimizer')
    # parser.add_argument("--num_sanity_val_steps", type=int, default=2, help='Sanity check runs n validation batches before starting the training routine')
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help='Gradient Accumulation')
    # parser.add_argument('--log_every_n_steps', type=int, default=100, help='How often to log metrics')

    # Others
    parser.add_argument('--xlsxfile_train', type=str, default="/network/scratch/b/bonaventure.dossou/probe_medical/reports/train/train_reports.csv", help='reports xlsx train file.')
    parser.add_argument('--xlsxfile_val', type=str, default="/network/scratch/b/bonaventure.dossou/probe_medical/reports/validation/validation_reports.csv", help='reports xlsx val file.')

    parser.add_argument('--trainfolder', type=str, default="/network/scratch/b/bonaventure.dossou/probe_medical/reports/train/data_volumes/dataset/train", help='train folder.')
    parser.add_argument('--validfolder', type=str, default="/network/scratch/b/bonaventure.dossou/probe_medical/reports/validation/data_volumes/dataset/valid", help='valid folder.')

    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    # parser.add_argument('--llama_model', type=str, default="epfl-llm/meditron-7b", help='LLM model to use.')

    args = parser.parse_args()
    return args

class DataModule(LightningDataModule):

    def __init__(
            self,
            args, train_dataset, dev_dataset
    ):
        super().__init__()
        self.args = args
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        """
        There are also data operations you might want to perform on every GPU. Use setup to do things like:

        count number of classes

        build vocabulary

        perform train/val/test splits

        apply transforms (defined explicitly in your datamodule or assigned in init)

        etc…
        :param stage:
        :return:
        """

        self.dataset = {
            "train": self.train_dataset, "validation": self.dev_dataset}


    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["train"], batch_size=self.args.batch_size, drop_last=True, pin_memory=True, shuffle=True,
            num_workers=self.args.num_workers)
        return loader


    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(self.dataset["validation"], batch_size=self.args.batch_size, drop_last=False, pin_memory=True, 
            shuffle=False, num_workers=self.args.num_workers)
        return loader


def main():
    # parse arguments
    args = parse_agrs()

    # create tokenizer
    tokenizer = Tokenizer(args)
    
    train_ds = CTReportDataset(args, data_folder=args.trainfolder, xlsx_file=args.xlsxfile_train, tokenizer=tokenizer, num_frames=2)
    valid_ds  = CTReportDataset(args, data_folder=args.validfolder, xlsx_file=args.xlsxfile_val, tokenizer=tokenizer, num_frames=2)

    print('Train Dataset: {}'.format(len(train_ds)))
    print('Valid Dataset: {}'.format(len(valid_ds)))

    # datamodule = DataModule(args, train_ds, valid_ds)

    # create data loader
    train_dataloader = R2DataLoader(args, train_ds, tokenizer, split='train', shuffle=True)
    val_dataloader = R2DataLoader(args, valid_ds, tokenizer, split='val', shuffle=False)

    # build model architecture
    model = CT2RepModel(args, tokenizer)

    # callbacks = add_callbacks(args)
    # checkpoint_callback = ModelCheckpoint(
    #     save_last=False,
    #     save_top_k=1,
    #     monitor="train_loss",
    #     mode="min",
    #     every_n_train_steps=args.every_n_train_steps,
    #     verbose=True,
    #     save_weights_only=False,
    #     dirpath="/network/scratch/b/bonaventure.dossou/probe_medical/models/checkpoints",
    #     filename="{epoch}-{step}-{train_loss}"
    # )

    # learning_rate_callback = callbacks["callbacks"][1]
    # trainer = pl.Trainer(
    #     devices=args.devices,
    #     num_nodes=args.num_nodes,
    #     strategy=args.strategy,
    #     accelerator=args.accelerator,
    #     precision=args.precision,
    #     val_check_interval = args.val_check_interval,
    #     limit_val_batches = args.limit_val_batches,
    #     max_epochs = args.max_epochs,
    #     num_sanity_val_steps = args.num_sanity_val_steps,
    #     accumulate_grad_batches=args.accumulate_grad_batches,
    #     callbacks=[checkpoint_callback, learning_rate_callback], # [callbacks["callbacks"], checkpoint_callback], 
    #     logger=callbacks["loggers"],
    #     log_every_n_steps=args.log_every_n_steps)

    # get function handles of loss and metrics
    criterion = compute_loss
    # metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    # metric_ftns = lambda gt, pred: {'bleu': metrics(gf, pred)}

    # model = CustomModel(model, criterion, optimizer, lr_scheduler, args)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, val_dataloader)
    trainer.train()

    # checkpoint_path = None
    # if checkpoint_path is None:
    #     trainer.fit(model, datamodule=datamodule)
    # else:
    #     trainer.fit(model, datamodule=datamodule, ckpt_path=checkpoint_path)
    # if args.validate:
    #     trainer.validate(model, datamodule=datamodule)


if __name__ == '__main__':
    # setup_distributed()
    main()
