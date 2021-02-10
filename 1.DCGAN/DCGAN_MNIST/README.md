# DCGAN_MNIST


## Your own data training

You collect images and put them to any directory.


## Preparation and configuation

# Assume that you have ccollect images and put them to any directory as mentioned in the README.md of the root directory

You may change the number of iteration  **Iteration** in **config.py**.
You may also set one or more absolute passes.
You may change directory path **Train_dirs** in **config.py**.
For example, 
```python
Train_dirs = [
    '/home/usrs/ADT-GAN/DATA/MNIST_PNG/MNIST-9'
]
```

### Usage

```bash
When training,
$ python main.py --train

When testing,
$ python main.py --test
```

The models are defined by **model.py**

Generated images in training process are stored in **train_images** (defined in **config.py**).

Generated images used to calculate the FID are stored in **all_images** (defined in **config.py**).