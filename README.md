## [Denoising based Sequence-to-Sequence Pre-training for Text Generation](https://arxiv.org/pdf/1908.08206.pdf) 


Implementation of EMNLP 2019 paper 
"[Denoising based Sequence-to-Sequence Pre-training for Text Generation](https://arxiv.org/pdf/1908.08206.pdf)",
available at [https://arxiv.org/abs/1908.08206](https://arxiv.org/abs/1908.08206).

PoDA is short for "**P**re-training **o**f **D**enoising **A**uto-encoders".

Our code is based on [fairseq@1d79ed9b5f67a51e468d](https://github.com/pytorch/fairseq/),
the major changes can be found at [commit cf7ca94a](https://github.com/yuantiku/PoDA/commit/cf7ca94aebbce1901ee90045cb023cce9a6e4412).

### Download pre-trained models

You can download the pre-trained model from

Google drive: [https://drive.google.com/open?id=1bW5e8287purr6U81s97-_EFMb35fxwUp](https://drive.google.com/open?id=1bW5e8287purr6U81s97-_EFMb35fxwUp), or

Baidu Cloud Disk: [https://pan.baidu.com/s/1D5tumkNRhT2pcSb3CqzE5w](https://pan.baidu.com/s/1D5tumkNRhT2pcSb3CqzE5w), code: w6j5


Unzip the pre-trained model to the folder `da-pretrained/`.

### Requirements

pytorch == 0.4 (Other versions of pytorch may also work, 0.4 is the version we use)

pyrouge 


### How to run

In `./demo_giga_1k/`,
we provide 1000 training examples from Gigaword dataset for demo.

Step 0, make sure you already downloaded the pre-trained model.

Step 1, preprocess and binarize the dataset

```bash
bash ./giga/preprocess_giga.sh &
```

Step 2, fine-tune the pre-trained model

```bash
bash ./giga/finetune.sh 0 test_run & 
```

Step 3, evaluate the model performance (in this case, the ROUGE score)

```bash
# run beam search decoding
bash ./giga/generate.sh out_giga/test_run/ 0 models_giga/model_test_run/checkpoint_best.pt &

# compute ROUGE score, please install pyrouge first
bash ./giga/eval_local.sh demo_giga_1k/test.trg out_giga/test_run/output.tok.txt rouge.log &
```

Please checkout the script to see what the argument means.

With only 1000 training examples,
you will get a reasonably good ROUGE score.
On my computer,
I get `ROUGE-1 Average_F = 0.289, ROUGE-2 Average_F = 0.110, ROUGE-L Average_F = 0.267 `,
which is almost even with results from [A Neural Attention Model for Sentence Summarization](https://www.aclweb.org/anthology/D15-1044)
trained on the entire 3.8 million examples.


### Outputs by PoDA

We provide the outputs by PoDA on four datasets in `predictions/`,
including [CNN/Daily Mail](https://github.com/abisee/cnn-dailymail), 
[Gigaword](https://github.com/harvardnlp/sent-summary), 
[CoNLL-2014](https://www.comp.nus.edu.sg/~nlp/conll14st.html), 
and [JFLEG](https://github.com/keisks/jfleg).


### Reproducibility

For CoNLL-2014 and JFLEG dataset,
we require spell error check as a preprocessing step,
which has dependency on our internal code.
Hopefully you can use/train a good public spell checker.

### Citation

If you find our paper or this repository helpful, 
please cite as follows:

```
@inproceedings{wang2019denoising,
  title={Denoising based Sequence-to-Sequence Pre-training for Text Generation},
  author={Wang, Liang and Zhao, Wei and Jia, Ruoyu and Li, Sujian and Liu, Jingming},
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},
  pages={3994--4006},
  year={2019}
}
```

### TODO

We will release the code for pre-training in a later time.
