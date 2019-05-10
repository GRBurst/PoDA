import os
import torch
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', default='./models/model_da/checkpoint_last.pt', help='path to model')
args = parser.parse_args()
assert os.path.exists(args.model), 'Model %s does not exist' % args.model

if __name__ == '__main__':
    lm_states = torch.load(args.model)
    for k in lm_states:
        print(k)
    del lm_states['args']
    del lm_states['last_optimizer_state']
    del lm_states['optimizer_history']
    del lm_states['extra_state']
    for k, v in lm_states['model'].items():
        print(k, v.size())
    torch.save(lm_states['model'], args.model + '.sm')
