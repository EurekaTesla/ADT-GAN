# ADT-GAN_MNIST


Before that, you need to pre-train on dataset MNIST-not9 with DCGAN, and Copy models G1 and D1 to **models**. 

We have saved the pre-trained models G1 and D1 that you can use directly.
Downloading Pre-trained_models: https://github.com/EurekaTesla/ADT-GAN
Copying **G1** and **D1** in **Pre-trained_models/Pre-trained_models_MNIST** to **ADT-GAN/3.ADT-GAN/ADT-GAN_MNIST/models**.



## Preparation and configuation

# Assume that you have ccollect images and put them to any directory as mentioned in the README.md of the root directory

You may change the number of iteration  **Iteration** in **config.py**.
You may also set one or more absolute passes.
You may change directory path **Train_dirs** in **config.py**.
For example, 
```python
Train_dirs = [
    '/home/usrs/ADT-GAN/DATA/MNIST_PNG/MNIST-9',
]
```

### Usage

```bash
For training,
  $ python main.py --train
  
For testing,
   $ python main.py --test
```

The models are defined by **model.py**

Generated images in training process are stored in **train_image** (defined in **config.py**).

Generated images used to calculate the FID are stored in **all_images** (defined in **config.py**).
