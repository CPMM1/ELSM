# Generative Recommenders

The source code of ``Enhancing Large Models based Sequential Recommendation with
Multimodal Graph Convolution Network``

Our code is implemented based on ``Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations``, [here](https://github.com/facebookresearch/generative-recommenders) is their project address.



## Getting started

#### Data preparation

To reproduce the results in our paper please download the processed pantry and scientific data from [here](https://drive.google.com/drive/folders/116QDSVlrsR6IvTR7_1Q5lkptSLzaM1bk?usp=sharing). Then place the downloaded ``tmp`` file in  home directory.

#### Run model training.

A GPU with 24GB or more HBM should work for most datasets.

```
CUDA_VISIBLE_DEVICES=0 python3 train.py --gin_config_file=configs/ml-1m/hstu-sampled-softmax-n128-large-final.gin --master_port=12345
```

Other configurations are included in configs/ml-1m, configs/ml-20m, and configs/amzn-books to make reproducing these experiments easier.



