# Author Toshihiko Aoki
#
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BertRegressionDataset for BERT."""

import json
import itertools
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from collections import defaultdict
from tokenization_sentencepiece import convert_tokens_to_ids

def tokenize(text, tokenizer, vocab):
    ids = convert_tokens_to_ids(vocab, tokenizer.tokenize(text))
    return ids

class RegressionDataset(Dataset):
    def __init__(
        self,
        tokenizer, max_pos, label_num=-1,
        dataset_path=None, delimiter='\t', encoding='utf-8', header_skip=True,
        sentence_a=[], sentence_b=[], labels=[],
        under_sampling=False
    ):
        super().__init__()
        self.targets = []
        self.text_records = []

        with open(dataset_path) as f:
            data = json.load(f)
        for item in data:
            self.targets += [item["targets"]]
            self.text_records += [tokenize(item["text"], tokenizer)]


    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return [torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.float) for x, y in zip(self.text_records[index], self.targets[index])]

    def per_label_record(self):
        return self.per_label_records_num

    def label_num(self):
        return len(self.per_label_records_num)

    def next_under_samples(self):
        if self.under_sample_num is None:
            return

        self.records = []
        current_pos = self.under_sample_num * self.sampling_index
        next_pos = self.under_sample_num * (self.sampling_index+1)
        for label_num in range(len(self.per_label_records)):
            origin_num = self.origin_per_label_records_num[label_num]

            if origin_num is self.under_sample_num:
                for sample in self.per_label_records[label_num]:
                    self.records.append(sample)
                continue

            if next_pos <= origin_num:
                for sample in self.per_label_records[label_num][current_pos-1:next_pos-1]:
                    self.records.append(sample)
                continue

            if current_pos < origin_num:
                next_num = origin_num - current_pos
                for sample in self.per_label_records[label_num][current_pos-1:current_pos-1 + next_num]:
                    self.records.append(sample)
                for sample in self.per_label_records[label_num][0: self.under_sample_num - next_num]:
                    self.records.append(sample)
                continue

            sample_mod = current_pos % origin_num
            if sample_mod == 0:
                for sample in self.per_label_records[label_num][0:self.under_sample_num]:
                    self.records.append(sample)
                continue

            if origin_num < (sample_mod - 1 + self.under_sample_num):
                add_pos = (sample_mod - 1 + self.under_sample_num) - origin_num
                for sample in self.per_label_records[label_num][sample_mod-1:origin_num]:
                    self.records.append(sample)
                for sample in self.per_label_records[label_num][0:add_pos]:
                    self.records.append(sample)
            else:
                for sample in self.per_label_records[label_num][sample_mod-1:sample_mod-1 + self.under_sample_num]:
                    self.records.append(sample)

        self.sampling_index += 1
