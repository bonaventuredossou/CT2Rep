import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import nibabel as nib
import tqdm

def cast_num_frames(t, *, frames):
    f = t.shape[1]
    if f%frames==0:
        return t[:,:-(frames-1)]
    if f%frames==1:
        return t
    else:
        return t[:,:-((f%frames)-1)]


class CTReportDataset(Dataset):
    def __init__(self, args, data_folder, xlsx_file, tokenizer, min_slices=20, resize_dim=500, num_frames=2, force_num_frames=True):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.tokenizer = tokenizer
        self.max_seq_length = args.max_seq_length
        self.accession_to_text = self.load_accession_text(xlsx_file)
        self.paths=[]
        self.samples = self.prepare_samples()

        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)
        self.cast_num_frames_fn = partial(cast_num_frames, frames = num_frames) if force_num_frames else identity

    def load_accession_text(self, xlsx_file):
        # df = pd.read_excel(xlsx_file)
        df = pd.read_csv(xlsx_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = str(row["Findings_EN"]).lower()
        return accession_to_text


    def prepare_samples(self):
        samples = []

        for patient_folder in tqdm.tqdm(glob.glob(os.path.join(self.data_folder, '*'))):            
            for accession_folder in glob.glob(os.path.join(patient_folder, '*')):
                # glob.glob(os.path.join(accession_folder, '*.npz'))
                for nii_file in glob.glob(os.path.join(accession_folder, '*.nii.gz')):
                    accession_number = nii_file.split("/")[-1]
                    # accession_number = accession_number.replace(".npz", ".nii.gz")
                    if accession_number not in self.accession_to_text:
                        continue

                    findings_text = self.accession_to_text[accession_number]
                    samples.append((nii_file, findings_text))
                    self.paths.append(nii_file)

        return samples

    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        try:
            img_data = np.load(path, allow_pickle=True)['arr_0']
        except:
            try:
                img_data = np.load(path, allow_pickle=True)["data"]
            except:
                nii_img = nib.load(str(path))
                img_data = nii_img.get_fdata()
    
        img_data= np.transpose(img_data, (1, 2, 0))
        img_data = img_data*1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data+400 ) / 600)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)

        # Get the dimensions of the input tensor
        target_shape = (480,480,240)
        
        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)

        tensor = tensor.permute(2, 0, 1)
        
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor[0]

    
    # def __getitem__(self, index):
    #     img, text = self.samples[index]
    #     image_id = img.split("/")[-1]
    #     tensor = self.nii_to_tensor(img)
    #     tokenized = self.tokenizer(text)
    #     reports_ids = tokenized['input_ids']
    #     reports_mask = tokenized['attention_mask']
    #     seq_lenght = len(ids)
    #     # sample = (img_id, tensor, ids, mask, seq_lenght)

    #     images = torch.stack(tensor, 0)
    #     targets = np.zeros((len(reports_ids), seq_lenght), dtype=int)

    #     for i, report_ids in enumerate(reports_ids):
    #         targets[i, :len(report_ids)] = report_ids

    #     return images_id, images, torch.LongTensor(targets), torch.FloatTensor(reports_mask)
    #     # return sample

    def __getitem__(self, index):
        img, text = self.samples[index]
        img_id = img.split("/")[-1]
        tensor = self.nii_to_tensor(img)
        ids = self.tokenizer(text)[:self.max_seq_length]
        mask = [1] * len(ids)
        seq_lenght = len(ids)
        sample = (img_id, tensor, ids, mask, seq_lenght)
        return sample