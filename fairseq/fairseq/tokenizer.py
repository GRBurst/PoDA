# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from collections import Counter
import re

import torch


SPACE_NORMALIZER = re.compile("\s+")


def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


class Tokenizer:

    @staticmethod
    def add_file_to_dictionary(filename, dict, tokenize):
        with open(filename, 'r') as f:
            for line in f:
                for word in tokenize(line):
                    dict.add_symbol(word)
                dict.add_symbol(dict.eos_word)

    @staticmethod
    def binarize(filename, dict, consumer, tokenize=tokenize_line,
                 append_eos=True, reverse_order=False, src_ids=None):
        nseq, ntok = 0, 0
        replaced = Counter()
        copied = Counter()

        def replaced_consumer(word, idx):
            if idx == dict.unk_index and word != dict.unk_word:
                replaced.update([word])
            if idx >= len(dict):
                copied.update([word])

        tokens_list = []
        with open(filename, 'r', encoding="utf-8") as f:
            for line in f:
                src_tokens = None
                if src_ids is not None:
                    src_tokens = src_ids[nseq]

                ids, tokens = Tokenizer.tokenize(
                    line=line,
                    dict=dict,
                    tokenize=tokenize,
                    add_if_not_exist=False,
                    consumer=replaced_consumer,
                    append_eos=append_eos,
                    reverse_order=reverse_order,
                    src_tokens=src_tokens
                )
                tokens_list.append(tokens)
                nseq += 1

                consumer(ids)
                ntok += len(ids)
        return {'nseq': nseq, 'ntok': ntok,
                'replaced': len(replaced), 'nunk': sum(replaced.values()),
                'copied': len(copied), 'ncopied': sum(copied.values())}, tokens_list

    @staticmethod
    def tokenize(line, dict, tokenize=tokenize_line, add_if_not_exist=True,
                 consumer=None, append_eos=True, reverse_order=False, src_tokens=None):
        words = tokenize(line)
        if reverse_order:
            words = list(reversed(words))
        nwords = len(words)
        ids = torch.IntTensor(nwords + 1 if append_eos else nwords)

        for i, word in enumerate(words):
            if add_if_not_exist:
                idx = dict.add_symbol(word)
            else:
                idx = dict.index(word)  # will return unk_idx if word not in dict
                if idx == dict.unk_index:
                    if src_tokens is not None:
                        if word in src_tokens:
                            position = src_tokens.index(word)
                            idx = position + len(dict)  # replace unk_index with the first position of word in src
                    else:
                        position = words.index(word)
                        idx = position + len(dict) # replace unk_index with the first position of word in src

            if consumer is not None:
                consumer(word, idx)
            ids[i] = idx
        if append_eos:
            ids[nwords] = dict.eos_index

        return ids, words
