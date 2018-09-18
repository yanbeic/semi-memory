# semi-memory

[Tensorflow](https://www.tensorflow.org/) Implementation of the paper Chen et al. [**Semi-Supervised Deep Learning with Memory, ECCV2018**](http://openaccess.thecvf.com/content_ECCV_2018/papers/Yanbei_Chen_Semi-Supervised_Deep_Learning_ECCV_2018_paper.pdf).


## Getting Started

### Prerequisite:

Tensorflow version >= 1.4.0.

### Data preparation:

1. Download and prepare datasets:

```
bash scripts/download_prepare_datasets.sh
```

2. Convert image data to tfrecords:

```
bash scripts/convert_images_to_tfrecords.sh
```


## Running Experiments

### Training & Testing:

For example, to train and test on `svhn`, run the following command.
```
bash scripts/train_svhn_semi.sh
```


## Citation

Please refer to the following if this repository is useful for your research.

### Bibtex:

```
@inproceedings{chen2018semi,
  title={Semi-Supervised Deep Learning with Memory},
  author={Chen, Yanbei and Zhu, Xiatian and Gong, Shaogang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

