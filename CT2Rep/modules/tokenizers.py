import json
import re
from collections import Counter
import pandas as pd
from transformers import LlamaTokenizer, LlamaForCausalLM

class Tokenizer(object):
    def __init__(self, args):
        self.threshold = args.threshold
        self.args = args
        self.pretrained_tokenizer = LlamaTokenizer.from_pretrained(args.llama_model)
        self.pretrained_model = LlamaForCausalLM.from_pretrained(args.llama_model)
        self.clean_report = self.clean_report_mimic_cxr
        self.accession_to_text = self.load_accession_text(args.xlsxfile)

        self.token2idx, self.idx2token = self.create_vocabulary()

        with open("idx2token.json", 'w') as json_file:
            # Write the dictionary to the file using JSON format
            json.dump(self.idx2token, json_file)

        with open("token2idx.json", 'w') as json_file:
            # Write the dictionary to the file using JSON format
            json.dump(self.token2idx, json_file)


    def load_accession_text(self, xlsx_file):
        df = pd.read_csv(xlsx_file[0])
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['AccessionNo']] = row["Findings_EN"].lower()
        return accession_to_text

    def create_vocabulary(self):
        total_tokens = []
        self.pretrained_tokenizer.pad_token = self.pretrained_tokenizer.eos_token

        for example in self.accession_to_text.values():
            tokens = self.clean_report(example).split()
            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + ['<unk>']
        vocab.sort()

        # token2idx, idx2token = {}, {}
        # for idx, token in enumerate(vocab):
        #     token2idx[token] = idx + 1
        #     idx2token[idx + 1] = token
        
        print('CT-RATE Dataset Tokens Size: {}'.format(len(vocab)))        

        # Check and filter new tokens
        existing_tokens = set(self.pretrained_tokenizer.get_vocab().keys())
        tokens_to_add = [token for token in vocab if token not in existing_tokens]

        # Add new tokens to the tokenizer
        if tokens_to_add:
            num_added_tokens = self.pretrained_tokenizer.add_tokens(tokens_to_add)
        else:
            num_added_tokens = 0

        # Resize model's embeddings if new tokens were added
        if num_added_tokens > 0:
            print('Adding {} new tokens to {} pretrained tokenizer'.format(num_added_tokens, self.args.llama_model))
            self.pretrained_model.resize_token_embeddings(len(self.pretrained_tokenizer))

            # Initialize new token embeddings
            new_token_ids = self.pretrained_tokenizer.convert_tokens_to_ids(tokens_to_add)
            existing_embeddings = self.pretrained_model.get_input_embeddings().weight.data
            new_embeddings = existing_embeddings.mean(dim=0, keepdim=True).repeat(len(tokens_to_add), 1)
            self.pretrained_model.get_input_embeddings().weight.data[new_token_ids] = new_embeddings

        token2idx = self.pretrained_tokenizer.get_vocab()
        idx2token = {idx: token for token, idx in token2idx.items()}

        return token2idx, idx2token

    def get_pretrained_components(self):
        return self.pretrained_model, self.pretrained_tokenizer

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
        tokens = ['<s>'] + self.clean_report(report).split() + ['</s>']
        ids = []
        for token in tokens:
            ids.append(self.get_id_by_token(token))
        # ids = [0] + ids + [0]
        # tokenized = self.pretrained_tokenizer(tokens, return_attention_mask=False, add_special_tokens=True)
        return ids # tokenized['input_ids'] # ids

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
