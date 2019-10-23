import os
import logging
import argparse
import random
import collections
from tqdm import tqdm, trange
import json


import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.modeling import BertModel

from tensorboardX import SummaryWriter

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, belief_input=None,labeld=None,labels=None,labelv=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            text_c: previious belief_input
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_c = belief_input
        self.text_a = text_a
        self.text_b = text_b

        self.labeld = labeld
        self.labels = labels
        self.labelv = labelv

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_len, label_id, segment_id,input_mask,labeld,labels,labelv,is_real_example=True,guid='NONE'):
        self.guid = guid
        self.input_ids = input_ids
        self.input_len = input_len
        self.label_id = label_id
        self.segment_id= segment_id
        self.input_mask = input_mask
        self.labeld = labeld
        self.labels = labels
        self.labelv = labelv
        self.is_real_example = is_real_example

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if len(line) > 0 and line[0][0] == '#':     # ignore comments (starting with '#')
                    continue
                lines.append(line)
            return lines


class Processor(DataProcessor):
    """Processor for the belief tracking dataset (GLUE version)."""

    def __init__(self, config):
        super(Processor, self).__init__()

        # MultiWOZ dataset

    def get_train_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "mwoz2_format_train.json"), accumulation)

    def get_dev_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "mwoz2_format_dev.json"),  accumulation)

    def get_test_examples(self, data_dir, accumulation=False):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "mwoz2_format_test.json"),  accumulation)


    def _create_examples(self, srcFile,  accumulation=False):
        """Creates examples for the training and dev sets."""
        srcF = open(srcFile, 'r')
        examples = []
        for i,l in enumerate(srcF):  # for each dialogue
            l = eval(l)
            guid = i
            # hierarchical input for a whole dialogue with multiple turns
            text_a = l['system_input']
            text_b = l['user_input']
            belief_input = l['belief_input']
            labeld = l['labeld']
            labels = l['labels']
            labelv = l['labelv']

            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, belief_input=belief_input,labeld=labeld,labels=labels,labelv=labelv))
        return examples

def _truncate_seq_pair(tokens, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_bert_input(tokens_a, tokens_b, belief_input, max_seq_length, tokenizer):
  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0

    dicts = {}
    dicts['src'] = tokenizer.vocab

    path = 'data/mwoz2_sl.dict'
    srcF = open(path, 'r')

    sl_dict = json.loads(srcF.read())
    srcF.close()

    path = 'data/mwoz2_dm.dict'
    srcF = open(path, 'r')
    dm_dict = json.loads(srcF.read())
    srcF.close()

    print(len(dicts['src']))

    print(sl_dict)
    print(dm_dict)

    for j in sl_dict.keys():
        for i in sl_dict[j].keys():
            if i not in dicts['src'].keys():
                dicts['src'][i] = len(dicts['src'])
                print(i)
                print(dicts['src'][i])

    for i in dm_dict.keys():
        if i not in dicts['src'].keys():
            dicts['src'][i] = len(dicts['src'])
            print(i)
            print(dicts['src'][i])

    print(len(dicts['src']))


    tokens = []
    segment_ids = []

    for token in belief_input:
        tokens.append(token)
        segment_ids.append(0)

    for token in tokens_a[1:]:
        tokens.append(token)
        segment_ids.append(1)


    for token in tokens_b[1:]:
        tokens.append(token)
        segment_ids.append(2)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    return tokens, input_ids, input_mask, segment_ids, dicts['src']





def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_turn_length):
    """Loads a data file into a list of `InputBatch`s."""

    for (ex_index, example) in enumerate(examples):

        input_text = example.text_c +example.text_a[1:]+example.text_b[1:]
        if example.text_b:
            tokens_b = [x if x != '#' else '[SEP]' for x in tokenizer.tokenize(example.text_b)]
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0      0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #

        tokens = tokens_c +tokens_a[1:] + tokens_b[1:]
        input_len = [len(tokens), 0]

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            input_len[1] = len(tokens_b) + 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        assert len(input_ids) == max_seq_length

        label_id = []
        label_info = 'label: '
        for i, label in enumerate(example.label):
            if label == 'dontcare':
                label = 'do not care'
            label_id.append(label_map[i][label])
            label_info += '%s (id = %d) ' % (label, label_map[i][label])

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_len: %s" % " ".join([str(x) for x in input_len]))
            logger.info("label: " + label_info)

        curr_dialogue_idx = example.guid.split('-')[1]
        curr_turn_idx = int(example.guid.split('-')[2])

        if prev_dialogue_idx is not None and prev_dialogue_idx != curr_dialogue_idx:
            if prev_turn_idx < max_turn_length:
                features += [InputFeatures(input_ids=all_padding,
                                           input_len=all_padding_len,
                                           label_id=[-1]*slot_dim)]\
                            *(max_turn_length - prev_turn_idx - 1)
            assert len(features) % max_turn_length == 0

        if prev_dialogue_idx is None or prev_turn_idx < max_turn_length:
            features.append(
                InputFeatures(input_ids=input_ids,
                              input_len=input_len,
                              label_id=label_id))

        prev_dialogue_idx = curr_dialogue_idx
        prev_turn_idx = curr_turn_idx

    if prev_turn_idx < max_turn_length:
        features += [InputFeatures(input_ids=all_padding,
                                   input_len=all_padding_len,
                                   label_id=[-1]*slot_dim)]\
                    * (max_turn_length - prev_turn_idx - 1)
    assert len(features) % max_turn_length == 0

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_len= torch.tensor([f.input_len for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    # reshape tensors to [#batch, #max_turn_length, #max_seq_length]
    all_input_ids = all_input_ids.view(-1, max_turn_length, max_seq_length)
    all_input_len = all_input_len.view(-1, max_turn_length, 2)
    all_label_ids = all_label_ids.view(-1, max_turn_length, slot_dim)

    return all_input_ids, all_input_len, all_label_ids


def main():
###############################################################################
# Test Load data
###############################################################################
# Get Processor
    processor = Processor(args)
    num_labels = [len(labels) for labels in label_list]  # number of slot-values in each slot-type

    # tokenizer
    vocab_dir = os.path.join(args.bert_dir, '%s-vocab.txt' % args.bert_model)
    if not os.path.exists(vocab_dir):
        raise ValueError("Can't find %s " % vocab_dir)
    tokenizer = BertTokenizer.from_pretrained(vocab_dir, do_lower_case=args.do_lower_case)

    num_train_steps = None
    accumulation = False