import argparse
import os
import json

from pyrouge import Rouge155

parser = argparse.ArgumentParser()
parser.add_argument('--test-trg', default='./processed_summ/eval/test/',
                    help='path to the ground-truth summaries')
parser.add_argument('--predict-trg', default='./processed_summ/eval/out_test/',
                    help='path to the model predicted summary')
parser.add_argument('--tmp-dir', default='./processed_summ/eval/out_test/',
                    help='path to the model predicted summary')
parser.add_argument('--eval-log', default='./test_score.log',
                    help='path to the evaluation score')


def make_html_safe(s):
    """Replace any angled brackets in string s to avoid interfering with HTML attention visualizer."""
    s.replace("<", "&lt;")
    s.replace(">", "&gt;")
    return s


def postprocess(line):
    words = line.split()
    ret = []
    for i, w in enumerate(words):
        if w == '<unk>' or w == 'UNK':
            ret.append('UNK')
        else:
            ret.append(w.lower())
    return ' '.join(ret)


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.exists(args.test_trg), '{} does not exist'.format(args.test_trg)
    assert os.path.exists(args.predict_trg), '{} does not exist'.format(args.predict_trg)

    os.system('rm -rf {}'.format(args.tmp_dir))
    system_dir = '{}/gold/'.format(args.tmp_dir)
    model_dir = '{}/predict/'.format(args.tmp_dir)
    os.makedirs(system_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    system_cnt = 0
    for line in open(args.test_trg, 'r', encoding='utf-8'):
        line = line.strip()
        system_cnt += 1
        with open('{}/{}.txt'.format(system_dir, system_cnt), 'w', encoding='utf-8') as w:
            w.write(line)

    predict_cnt = 0
    for line in open(args.predict_trg, 'r', encoding='utf-8'):
        line = postprocess(make_html_safe(line.strip()))
        if len(line) == 0:
            line = 'UNK'
        predict_cnt += 1
        with open('{}/{}.txt'.format(model_dir, predict_cnt), 'w', encoding='utf-8') as w:
            w.write(line)
    assert predict_cnt == system_cnt, 'Mismatch predict_cnt={}, system_cnt={}'.format(predict_cnt, system_cnt)

    r = Rouge155()
    r.system_dir = system_dir
    r.model_dir = model_dir
    r.system_filename_pattern = '([\d]+).txt'
    r.model_filename_pattern = '#ID#.txt'

    output = r.convert_and_evaluate()
    print(output)
    output_dict = r.output_to_dict(output)
    print('Evaluation results: {}'.format(json.dumps(output_dict)))

    with open(args.eval_log, 'w', encoding='utf-8') as f:
        f.write('{}\n'.format(output))
        f.write('{}\n'.format(json.dumps(output_dict)))

