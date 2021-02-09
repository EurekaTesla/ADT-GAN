# FID
The FID is the performance measure used to evaluate the experiments in the paper.

## Usage
from 2 image foldername/

CPU
```bash
$ python fid_official_tf.py /path/to/real_images/foldername1/ /path/to/generated_images/foldername2/ 
```
For example,
```bash
$ python fid_official_tf.py '/home/usrs/ADT-GAN/DATA/MNIST_PNG/MNIST-9' '/home/usrs/ADT-GAN//1.DCGAN/DCGAN_MNIST/all_images/200'
```

GPU 0/1/2...
```bash
$ python fid_official_tf.py /path/to/real_images/foldername1/ /path/to/generated_images/foldername2/ --gpu 0
```
For example,
```bash
$ python fid_official_tf.py '/home/usrs/ADT-GAN/DATA/MNIST_PNG/MNIST-9' '/home/usrs/ADT-GAN//1.DCGAN/DCGAN_MNIST/all_images/200' --gpu 0
```
