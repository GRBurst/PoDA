# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import contextlib
import itertools
import os

import numpy as np
import torch

from . import FairseqDataset, LanguagePairDatasetMerge


def infer_language_pair(path):
    """Infer language pair from filename: <split>.<lang1>-<lang2>.(...).idx"""
    src, dst = None, None
    for filename in os.listdir(path):
        parts = filename.split('.')
        if len(parts) >= 3 and len(parts[1].split('-')) == 2:
            return parts[1].split('-')
    return src, dst


class ShardedIterator(object):
    """A sharded wrapper around an iterable (padded to length)."""

    def __init__(self, iterable, num_shards, shard_id, fill_value=None):
        if shard_id < 0 or shard_id >= num_shards:
            raise ValueError('shard_id must be between 0 and num_shards')

        self._sharded_len = len(iterable) // num_shards
        if len(iterable) % num_shards > 0:
            self._sharded_len += 1

        self.itr = itertools.zip_longest(
            range(self._sharded_len),
            itertools.islice(iterable, shard_id, len(iterable), num_shards),
            fillvalue=fill_value,
        )

    def __len__(self):
        return self._sharded_len

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.itr)[1]


class CountingIterator(object):
    """Wrapper around an iterable that maintains the iteration count."""

    def __init__(self, iterable):
        self.iterable = iterable
        self.count = 0
        self.itr = iter(self)

    def __len__(self):
        return len(self.iterable)

    def __iter__(self):
        for x in self.iterable:
            self.count += 1
            yield x

    def __next__(self):
        return next(self.itr)

    def has_next(self):
        return self.count < len(self)

    def skip(self, num_to_skip):
        next(itertools.islice(self.itr, num_to_skip, num_to_skip), None)
        return self


def collate_tokens(values, pad_idx, eos_idx, left_pad, move_eos_to_beginning=False):
    """Convert a list of 1d tensors into a padded 2d tensor."""
    size = max(v.size(0) for v in values)
    res = values[0].new(len(values), size).fill_(pad_idx)

    def copy_tensor(src, dst):
        assert dst.numel() == src.numel()
        if move_eos_to_beginning:
            assert src[-1] == eos_idx
            dst[0] = eos_idx
            dst[1:] = src[:-1]
        else:
            dst.copy_(src)

    for i, v in enumerate(values):
        copy_tensor(v, res[i][size - len(v):] if left_pad else res[i][:len(v)])
    return res


class EpochBatchIterator(object):
    """Iterate over a FairseqDataset and yield batches bucketed by size.

    Batches may contain sequences of different lengths. This iterator can be
    reused across multiple epochs with the next_epoch_itr() method.

    Args:
        dataset: a FairseqDataset
        max_tokens: max number of tokens in each batch
        max_sentences: max number of sentences in each batch
        max_positions: max sentence length supported by the model
        ignore_invalid_inputs: don't raise Exception for sentences that are too long
        required_batch_size_multiple: require batch size to be a multiple of N
        seed: seed for random number generator for reproducibility
        num_shards: shard the data iterator into N shards
        shard_id: which shard of the data iterator to return
    """

    def __init__(
        self, dataset, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1, seed=1,
        num_shards=1, shard_id=0,
    ):
        assert isinstance(dataset, FairseqDataset)
        self.dataset = dataset
        self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
        self.max_sentences = max_sentences if max_sentences is not None else float('Inf')
        self.max_positions = max_positions
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.bsz_mult = required_batch_size_multiple
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id

        with numpy_seed(self.seed):
            self.frozen_batches = tuple(self._batch_generator(self.dataset, 0))

        self.epoch = 0
        self._cur_epoch_itr = None
        self._next_epoch_itr = None

    def __len__(self):
        return len(self.frozen_batches)

    def next_epoch_itr(self, shuffle=True):
        """Shuffle batches and return a new iterator over the dataset."""
        if self._next_epoch_itr is not None:
            self._cur_epoch_itr = self._next_epoch_itr
            self._next_epoch_itr = None
        else:
            self.epoch += 1
            self._cur_epoch_itr = self._get_iterator_for_epoch(self.epoch, shuffle)
        return self._cur_epoch_itr

    def end_of_epoch(self):
        return not self._cur_epoch_itr.has_next()

    @property
    def iterations_in_epoch(self):
        if self._cur_epoch_itr is not None:
            return self._cur_epoch_itr.count
        elif self._next_epoch_itr is not None:
            return self._next_epoch_itr.count
        return 0

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'iterations_in_epoch': self.iterations_in_epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict['epoch']
        itr_pos = state_dict.get('iterations_in_epoch', 0)
        if itr_pos > 0:
            # fast-forward epoch iterator
            itr = self._get_iterator_for_epoch(self.epoch, state_dict.get('shuffle', True))
            if itr_pos < len(itr):
                self._next_epoch_itr = itr.skip(itr_pos)

    def _get_iterator_for_epoch(self, epoch, shuffle):
        if shuffle:
            # set seed based on the seed and epoch number so that we get
            # reproducible results when resuming from checkpoints
            with numpy_seed(self.seed + epoch):
                batches = list(self.frozen_batches)  # copy
                np.random.shuffle(batches)
        else:
            batches = self.frozen_batches
        return CountingIterator(torch.utils.data.DataLoader(
            self.dataset,
            collate_fn=self.dataset.collater,
            batch_sampler=ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[]),
        ))

    def _batch_generator(self, dataset, start_idx):
        batch = []

        def is_batch_full(num_tokens):
            if len(batch) == 0:
                return False
            if len(batch) == self.max_sentences:
                return True
            if num_tokens > self.max_tokens:
                return True
            return False

        sample_len = 0
        sample_lens = []
        ignored = []
        for idx in dataset.ordered_indices():
            if not dataset.valid_size(idx, self.max_positions):
                if self.ignore_invalid_inputs:
                    ignored.append(idx + start_idx)
                    continue
                raise Exception((
                    'Size of sample #{} is invalid, max_positions={}, skip this '
                    'example with --skip-invalid-size-inputs-valid-test'
                ).format(idx + start_idx, self.max_positions))

            sample_lens.append(dataset.num_tokens(idx))
            sample_len = max(sample_len, sample_lens[-1])
            num_tokens = (len(batch) + 1) * sample_len
            if is_batch_full(num_tokens):
                mod_len = max(
                    self.bsz_mult * (len(batch) // self.bsz_mult),
                    len(batch) % self.bsz_mult,
                )
                yield batch[:mod_len]
                batch = batch[mod_len:]
                sample_lens = sample_lens[mod_len:]
                sample_len = max(sample_lens) if len(sample_lens) > 0 else 0

            batch.append(idx + start_idx)

        if len(batch) > 0:
            yield batch

        if len(ignored) > 0:
            print((
                '| WARNING: {} samples have invalid sizes and will be skipped, '
                'max_positions={}, first few sample ids={}'
            ).format(len(ignored), self.max_positions, ignored[:10]))


@contextlib.contextmanager
def numpy_seed(seed):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)


class EpochBatchIteratorDynamic(EpochBatchIterator):

    """Epoch batch iterator that supports dataset dict."""
    def __init__(
        self, dataset_dict, max_tokens=None, max_sentences=None, max_positions=None,
        ignore_invalid_inputs=False, required_batch_size_multiple=1, seed=1,
        num_shards=1, shard_id=0,
    ):
        self.dataset_dict = dataset_dict
        self.dataset_merge = LanguagePairDatasetMerge(dataset_dict)
        self.max_tokens = max_tokens if max_tokens is not None else float('Inf')
        self.max_sentences = max_sentences if max_sentences is not None else float('Inf')
        self.max_positions = max_positions
        self.ignore_invalid_inputs = ignore_invalid_inputs
        self.bsz_mult = required_batch_size_multiple
        self.seed = seed
        self.num_shards = num_shards
        self.shard_id = shard_id
        
        self.dataset_idx_range = {}
        self.batch_count = 0
        with numpy_seed(self.seed):
            self.frozen_batches_list = []
            self.dynamic_batches_list = []
            start_idx = 0
            for k in sorted(self.dataset_dict.keys()):
                batches = self._batch_generator(self.dataset_dict[k], start_idx=start_idx)
                batches = tuple(batches)
                self.batch_count += len(batches)

                n_repeat = self.dataset_dict[k].n_repeat
                batches_repeated = []
                for r in range(int(n_repeat)):
                    batches_repeated.extend(batches)
                if n_repeat % 1 > 0:
                    self.dynamic_batches_list.append((batches, n_repeat % 1))
                
                self.frozen_batches_list.append(batches_repeated)
                
                self.dataset_idx_range[k] = (start_idx, start_idx + len(self.dataset_dict[k]))  # [start, end)
                start_idx += len(self.dataset_dict[k])
                print("{} dataset, n_repeat={}, start_index={}, frozen_batches={}, random_repeat={}".format(k, n_repeat, start_idx, len(batches_repeated), n_repeat % 1))
            print('all dataset, end_index={}'.format(start_idx))

        self.epoch = 0
        self._cur_epoch_itr = None
        self._next_epoch_itr = None

    def __len__(self):
        return self.batch_count

    def _get_iterator_for_epoch(self, epoch, shuffle):
        batches = []
        for fb in self.frozen_batches_list:
            batches.extend(list(fb))

        # dynamic batches
        for dbs, n_repeat in self.dynamic_batches_list:
            batches_repeated = []
            for db in dbs:
                if np.random.rand() < n_repeat % 1:
                    batches_repeated.append(db)
            batches.extend(batches_repeated)

        if shuffle:
            # set seed based on the seed and epoch number so that we get
            # reproducible results when resuming from checkpoints
            with numpy_seed(self.seed + epoch):
                np.random.shuffle(batches)

        return CountingIterator(torch.utils.data.DataLoader(
            self.dataset_merge,
            collate_fn=self.dataset_merge.collater,
            batch_sampler=ShardedIterator(batches, self.num_shards, self.shard_id, fill_value=[]),
        ))
