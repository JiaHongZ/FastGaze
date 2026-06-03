# FastGaze

This repository contains the official implementation of **FastGaze: An Efficient and Flexible Model for Human Scanpath Prediction**.

## Datasets

The dataset package `FastGaze-cocosearch.zip` can be downloaded from Baidu Netdisk:

- Link: https://pan.baidu.com/s/1MO8NEjotY9WTnO0kjDaJ_Q?pwd=kqis
- Extraction code: `kqis`

After downloading and extracting the dataset, please update the data paths in the code according to your local directory structure.

## Pretrained Weights

The pretrained weights package `pretrain.zip` can be downloaded from the same Baidu Netdisk link:

- Link: https://pan.baidu.com/s/1MO8NEjotY9WTnO0kjDaJ_Q?pwd=kqis
- Extraction code: `kqis`

Please place the `.pkg` files in the corresponding directories. For example:

```bash
fv/fastgaze-T.pkg
```

should be placed under:

```bash
FastGaze_sep/model_zoo/
```

## Usage

Training, testing, and visualization commands are provided in the corresponding folders. Please refer to the instructions in each folder for details.

## Citation

If you find this repository useful for your research, please consider citing our paper:

```bibtex
@article{zhang2026fastgaze,
  title={FastGaze: An Efficient and Flexible Model for Human Scanpath Prediction},
  author={Zhang, Jiahong and Shen, Kai and Pei, Hongjuan and Hu, Tianxiang and Wang, Kexin and Shang, Di and Xu, Bo and Li, Guoqi},
  journal={Expert Systems with Applications},
  pages={132843},
  year={2026},
  publisher={Elsevier}
}
```
