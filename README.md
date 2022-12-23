# RE-TaMM

This is the implementation of [Relation Extraction with Type-aware Map Memories of Word Dependencies](https://aclanthology.org/2021.findings-acl.221.pdf) at ACL 2021.

You can e-mail Yuanhe Tian at `yhtian@uw.edu`, if you have any questions.


**Visit our [homepage](https://github.com/synlp/.github) to find more our recent research and softwares for NLP (e.g., pre-trained LM, POS tagging, NER, sentiment analysis, relation extraction, datasets, etc.).**

## Upgrades of RE-TaMM

We are improving our RE-TaMM. For updates, please visit [HERE](https://github.com/synlp/RE-TaMM).

## Citation

If you use or extend our work, please cite our paper at ACL 2021.

```
@article{chen2021relation,
  title={Relation Extraction with Type-aware Map Memories of Word Dependencies},
  author={Chen, Guimin and Tian, Yuanhe and Song, Yan and Wan, Xiang},
  journal={Findings of the Association for Computational Linguistics: ACLIJCNLP},
  year={2021}
}
```

## Requirements

Our code works with the following environment.
* `python>=3.7`
* `pytorch>=1.3`

## Dataset

To obtain the data, you can go to [`data`](./data) directory for details.

## Downloading BERT and RE-TaMM

In our paper, we use BERT ([paper](https://www.aclweb.org/anthology/N19-1423/)) as the encoder.

For BERT, please download pre-trained BERT-Base and BERT-Large English from [Google](https://github.com/google-research/bert) or from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz). If you download it from Google, you need to convert the model from TensorFlow version to PyTorch version.

For RE-TAMM, you can download the models we trained in our experiments from [Google Drive](https://drive.google.com/drive/folders/1NqN2S9VGbgmD6Z-V2YVncA9lHOaffOuM?usp=sharing).

## Run on Sample Data

Run `run_sample.sh` to train a model on the small sample data under the `sample_data` directory.

## Training and Testing

You can find the command lines to train and test models in `run_train.sh` and `run_test.sh`, respectively.

Here are some important parameters:

* `--do_train`: train the model.
* `--do_eval`: test the model.

## To-do List

* Regular maintenance.

You can leave comments in the `Issues` section, if you want us to implement any functions.

