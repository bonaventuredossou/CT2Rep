# import json
# import re
# from collections import Counter
# import pandas as pd
# from transformers import LlamaTokenizer, LlamaForCausalLM
# import torch

# class CustomTokenizer:
#     def __init__(self, vocab):
#         self.vocab = vocab
#         self.token_to_id = {token: idx for idx, token in enumerate(vocab)}
#         self.id_to_token = {idx: token for idx, token in enumerate(vocab)}
#         self.pretrained_vocab = None

#     def add_pretrained_tokenizer(self, pretrained_vocab):
#         self.pretrained_vocab = pretrained_vocab

#     def encode(self, text):
#         tokens = text.split()  # Simple whitespace tokenizer; adjust as needed
#         return [self.token_to_id[token] for token in tokens if token in self.token_to_id]

#     def decode(self, ids):
#         return ' '.join([self.id_to_token[id] for id in ids])

#     def __len__(self):
#         return len(self.vocab)


# class Tokenizer(object):
#     def __init__(self, args):
#         self.threshold = args.threshold
#         self.args = args
#         # self.pretrained_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model, cache_dir="/network/scratch/b/bonaventure.dossou")
#         # self.pretrained_model = LlamaForCausalLM.from_pretrained(args.llama_model, cache_dir="/network/scratch/b/bonaventure.dossou", torch_dtype=torch.float32)
#         self.clean_report = self.clean_report_mimic_cxr
#         self.accession_to_text = self.load_accession_text()
#         self.tokenizer = None
#         self.token2idx, self.idx2token = self.create_vocabulary()

#         with open("idx2token.json", 'w') as json_file:
#             # Write the dictionary to the file using JSON format
#             json.dump(self.idx2token, json_file)

#         with open("token2idx.json", 'w') as json_file:
#             # Write the dictionary to the file using JSON format
#             json.dump(self.token2idx, json_file)


#     def load_accession_text(self):
#         train_file = self.args.xlsxfile_train
#         valid_file = self.args.xlsxfile_val

#         df_train = pd.read_csv(train_file)
#         df_val = pd.read_csv(valid_file)

#         accession_to_text = {}
#         for index, row in df_train.iterrows():
#             accession_to_text[row['VolumeName']] = str(row["Findings_EN"]).lower()

#         for index, row in df_val.iterrows():
#             accession_to_text[row['VolumeName']] = str(row["Findings_EN"]).lower()

#         return accession_to_text

#     def create_vocabulary(self):
#         total_tokens = []
#         # self.pretrained_tokenizer.pad_token = self.pretrained_tokenizer.eos_token

#         for example in self.accession_to_text.values():
#             tokens = self.clean_report(example).split()
#             for token in tokens:
#                 total_tokens.append(token)

#         counter = Counter(total_tokens)
#         # eos_token = self.pretrained_tokenizer.eos_token
#         # bos_token = self.pretrained_tokenizer.bos_token
#         # unk_token = self.pretrained_tokenizer.unk_token
#         # pad_token = self.pretrained_tokenizer.pad_token

#         vocab = [k for k, v in counter.items() if v >= self.threshold]
#         vocab.sort()
#         # vocab = [eos_token, bos_token, unk_token] + vocab # eos == padding

#         print('CT-RATE Dataset Tokens Size: {}'.format(len(vocab)))
#         # print('CT-RATE Dataset Special Tokens: {}'.format(vocab[:3]))        

#         # Add new tokens to the tokenizer
#         # num_added_tokens = self.pretrained_tokenizer.add_tokens(vocab, special_tokens=True)

#         # # Resize model's embeddings if new tokens were added
#         # if num_added_tokens > 0:
#         #     print('Adding {} new tokens to {} pretrained tokenizer'.format(num_added_tokens, self.args.llama_model))
#         #     # Extract the original embeddings
#         #     original_embeddings = self.pretrained_model.get_input_embeddings().weight.data
#         #     # Create new embeddings for the selected tokens
#         #     new_embeddings = torch.zeros((len(vocab), self.pretrained_model.config.hidden_size))
#         #     # Map the selected token embeddings to the new embeddings matrix
#         #     for idx, token in enumerate(vocab):
#         #         # we need to make sure we take the token that exist in the original embeddings
#         #         try:
#         #             if token in self.pretrained_tokenizer.get_vocab():                    
#         #                 new_embeddings[idx] = original_embeddings[self.pretrained_tokenizer.convert_tokens_to_ids(token)]
#         #         except:
#         #             pass
#         #     # we resize the model embedding to the size of the new dictionary
#         #     self.pretrained_model.resize_token_embeddings(len(vocab))
#         #     # Assign the new embeddings to the model
#         #     self.pretrained_model.get_input_embeddings().weight = torch.nn.Parameter(new_embeddings)
#         #     # Adjust the output layer of the model to match the new vocabulary size
#         #     self.pretrained_model.lm_head = torch.nn.Linear(self.pretrained_model.config.hidden_size, len(vocab), bias=False)
#         #     self.pretrained_model.tie_weights()

#         # ct_tokenizer = CustomTokenizer(vocab)        
#         # pretrained_dict = {self.pretrained_tokenizer.convert_tokens_to_ids(token): idx for idx, token in enumerate(vocab)}
#         # ct_tokenizer.add_pretrained_tokenizer(pretrained_dict)
#         # self.tokenizer = ct_tokenizer

#         token2idx = self.tokenizer.token_to_id
#         idx2token = self.tokenizer.id_to_token

#         return token2idx, idx2token

#     def get_pretrained_components(self):
#         return self.pretrained_model, self.pretrained_tokenizer

#     def clean_report_iu_xray(self, report):
#         report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
#             .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
#             .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
#                                         replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report

#     def clean_report_mimic_cxr(self, report):
#         report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
#             .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
#             .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
#             .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
#             .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
#             .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
#             .strip().lower().split('. ')
#         sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
#                                         .replace('\\', '').replace("'", '').strip().lower())
#         tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
#         report = ' . '.join(tokens) + ' .'
#         return report

#     def get_token_by_id(self, id):
#         return self.idx2token[id]

#     def get_id_by_token(self, token):
#         if token not in self.token2idx:
#             return self.token2idx['<unk>']
#         return self.token2idx[token]

#     def get_vocab_size(self):
#         return len(self.token2idx)

#     def __call__(self, report):
#         cleaned_report = self.clean_report(report)
#         tokenized = self.pretrained_tokenizer(cleaned_report, max_length=self.args.max_seq_length,
#             truncation=True, padding="max_length", return_attention_mask=True, add_special_tokens=True)
        
#         input_ids_pretrained = tokenized['input_ids']
#         input_ids_mapped = [self.tokenizer.pretrained_vocab[pretrained_token] for pretrained_token in input_ids_pretrained]

#         # we need to map the pretrained tokenizer tokens to the token of the customm tokenizer
#         remapped_tokenizer = {'input_ids': input_ids_mapped, 'attention_mask': tokenized['attention_mask']}
#         del tokenized
#         return remapped_tokenizer

#     def decode(self, ids):
#         return self.tokenizer.decode(ids)

#     def decode_batch(self, ids_batch):
#         out = []
#         for ids in ids_batch:
#             out.append(self.decode(ids))
#         return out


import json
import re
from collections import Counter
import pandas as pd

class Tokenizer(object):
    def __init__(self, args):
        self.threshold = args.threshold
        self.clean_report = self.clean_report_mimic_cxr
        self.accession_to_text = self.load_accession_text()

        self.token2idx, self.idx2token = self.create_vocabulary()

        with open("idx2token.json", 'w') as json_file:
            # Write the dictionary to the file using JSON format
            json.dump(self.idx2token, json_file)

        with open("token2idx.json", 'w') as json_file:
            # Write the dictionary to the file using JSON format
            json.dump(self.token2idx, json_file)


    def load_accession_text(self):
        train_file = self.args.xlsxfile_train
        valid_file = self.args.xlsxfile_val

        df_train = pd.read_csv(train_file)
        df_val = pd.read_csv(valid_file)

        accession_to_text = {}
        for index, row in df_train.iterrows():
            accession_to_text[row['VolumeName']] = str(row["Findings_EN"]).lower()

        for index, row in df_val.iterrows():
            accession_to_text[row['VolumeName']] = str(row["Findings_EN"]).lower()

        return accession_to_text

    def create_vocabulary(self):
        total_tokens = []

        for example in self.accession_to_text.values():

            tokens = self.clean_report(example).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()
        token2idx, idx2token = {}, {}
        for idx, token in enumerate(vocab):
            token2idx[token] = idx + 1
            idx2token[idx + 1] = token
        return token2idx, idx2token

    def clean_report_iu_xray(self, report):
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def clean_report_mimic_cxr(self, report):
        report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
            .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
            .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
            .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
            .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                        .replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        ids = [0] + ids + [0]
        return ids

    def decode(self, ids):
        txt = ''
        for i, idx in enumerate(ids):
            if idx > 0:
                if i >= 1:
                    txt += ' '
                txt += self.idx2token[idx]
            else:
                break
        return txt

    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out