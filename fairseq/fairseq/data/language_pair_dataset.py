# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import numpy as np
import torch

from . import FairseqDataset


def collate(samples, pad_idx, eos_idx, unk_idx, dict_size, left_pad_source=True, left_pad_target=False, 
        use_copy=False):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        from .data_utils import collate_tokens
        return collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    name = samples[0]['name']
    assert(samples[-1]['name'] == name)

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([s['source'].numel() for s in samples])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        # we create a shifted version of targets for feeding the
        # previous output token(s) into the next decoder step
        prev_output_tokens = merge(
            'target',
            left_pad=left_pad_target,
            move_eos_to_beginning=True,
        )
        prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
        target = target.index_select(0, sort_order)
        ntokens = sum(len(s['target']) for s in samples)
    else:
        ntokens = sum(len(s['source']) for s in samples)

    if not use_copy:
        src_tokens = src_tokens.masked_fill(src_tokens >= dict_size, unk_idx)
        if prev_output_tokens is not None:
            target = target.masked_fill(target >= dict_size, unk_idx)
            prev_output_tokens = prev_output_tokens.masked_fill(prev_output_tokens >= dict_size, unk_idx)

    return {
        'id': id,
        'ntokens': ntokens,
        'name': name,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
            'prev_output_tokens': prev_output_tokens,
        },
        'target': target,
    }


class LanguagePairDataset(FairseqDataset):
    """A pair of torch.utils.data.Datasets."""

    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        left_pad_source=True, left_pad_target=False,
        max_source_positions=1024, max_target_positions=1024,
        shuffle=True, use_copy=False, n_repeat=1, name='default'
    ):
        if tgt_dict is not None:
            assert src_dict.pad() == tgt_dict.pad()
            assert src_dict.eos() == tgt_dict.eos()
            assert src_dict.unk() == tgt_dict.unk()
        self.src = src
        self.tgt = tgt
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.shuffle = shuffle
        self.use_copy = use_copy
        self.n_repeat = n_repeat
        self.name = name

        self.n_special = src_dict.nspecial
        self.word_dict_size = len(src_dict)

    def __getitem__(self, index):
        return {
            'id': index,
            'name': self.name,
            'source': self.src[index],
            'target': self.tgt[index] if self.tgt is not None else None,
        }

    def __len__(self):
        return len(self.src)

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(
            samples, pad_idx=self.src_dict.pad(), eos_idx=self.src_dict.eos(),
            unk_idx=self.src_dict.unk(), dict_size=len(self.src_dict),
            left_pad_source=self.left_pad_source, left_pad_target=self.left_pad_target,
            use_copy=self.use_copy
        )

    def get_dummy_batch(self, num_tokens, max_positions, src_len=128, tgt_len=128):
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        src_len, tgt_len = min(src_len, max_source_positions), min(tgt_len, max_target_positions)
        bsz = num_tokens // max(src_len, tgt_len)
        return self.collater([
            {
                'name': 'dummy',
                'id': i,
                'source': self.src_dict.dummy_sentence(src_len),
                'target': self.tgt_dict.dummy_sentence(tgt_len) if self.tgt_dict is not None else None,
            }
            for i in range(bsz)
        ])

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        return max(self.src_sizes[index], self.tgt_sizes[index] if self.tgt_sizes is not None else 0)

    def ordered_indices(self):
        """Ordered indices for batching."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.tgt_sizes[indices], kind='mergesort')]
        return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]

    def valid_size(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        max_source_positions, max_target_positions = self._get_max_positions(max_positions)
        return (
            self.src_sizes[index] <= max_source_positions
            and (self.tgt_sizes is None or self.tgt_sizes[index] <= max_target_positions)
        )

    def _get_max_positions(self, max_positions):
        if max_positions is None:
            return self.max_source_positions, self.max_target_positions
        assert len(max_positions) == 2
        max_src_pos, max_tgt_pos = max_positions
        return min(self.max_source_positions, max_src_pos), min(self.max_target_positions, max_tgt_pos)


class LanguagePairDatasetMerge(LanguagePairDataset):
    def __init__(self, dataset_dict, name='merge'):
        data_list = []
        lens = [0]
        for k in sorted(dataset_dict.keys()):
            data_list.append(dataset_dict[k])
            lens.append(lens[-1] + len(dataset_dict[k]))

        self.data_list = data_list
        self.lens = lens
        self.name = name

    def _get_real_index(self, index):
        for i, ll in enumerate(self.lens[1:]):
            if index < ll:
                return i, index - self.lens[i]
        raise Exception('Invalid index')

    def __getitem__(self, index):
        i, index = self._get_real_index(index)
        return self.data_list[i].__getitem__(index)

    def __len__(self):
        return self.lens[-1]

    def collater(self, samples):
        return self.data_list[0].collater(samples)

    def num_tokens(self, index):
        """Return an example's length (number of tokens), used for batching."""
        i, index = self._get_real_index(index)
        return max(self.data_list[i].src_sizes[index], self.data_list[i].tgt_sizes[index] if self.tgt_sizes is not None else 0)
    
    def valid_size_dummy(self, index, max_positions):
        """Check if an example's size is valid according to max_positions."""
        
        i, index = self._get_real_index(index)
        
        max_source_positions, max_target_positions = self.data_list[i]._get_max_positions(max_positions)
        return (
            self.data_list[i].src_sizes[index] <= max_source_positions
            and (self.data_list[i].tgt_sizes is None or self.data_list[i].tgt_sizes[index] <= max_target_positions)
        )
