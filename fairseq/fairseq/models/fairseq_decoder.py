# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch
import torch.nn as nn
import torch.nn.functional as F


class FairseqDecoder(nn.Module):
    """Base class for decoders."""

    def __init__(self, dictionary):
        super().__init__()
        self.dictionary = dictionary

    def forward(self, prev_output_tokens, encoder_out):
        raise NotImplementedError

    def get_normalized_probs(self, net_output, log_probs, _):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output['decoder_out']
        copy_scores = net_output['copy_scores']

        if copy_scores is not None:
            alpha = net_output['copy_alpha']
            src_tokens = net_output['src_tokens']
            src_len = src_tokens.size(1)
            incre = (len(logits.size()) == 2)
            if incre:
                logits = logits.unsqueeze(1)
            
            src_tokens = src_tokens.unsqueeze(1).repeat(1, logits.size(1), 1)
            
            logits = F.softmax(logits, dim=-1)
            dummy_logits = torch.zeros(logits.size(0), logits.size(1), src_len).cuda()
            logits = torch.cat([logits, dummy_logits], dim=-1).float()

            logits = alpha * logits
            copy_scores = (1 - alpha) * copy_scores.float()
            logits.scatter_add_(dim=-1, index=src_tokens, src=copy_scores)

            if incre:
                logits = logits.squeeze()

            if log_probs:
                result = torch.log(logits + 1e-12)
            else:
                result = logits
        elif log_probs:
            result = F.log_softmax(logits.float(), dim=-1)
        else:
            result = F.softmax(logits.float(), dim=-1)

        return result

    def max_positions(self):
        """Maximum input length supported by the decoder."""
        raise NotImplementedError

    def upgrade_state_dict(self, state_dict):
        return state_dict
