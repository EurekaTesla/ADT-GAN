# ADT-GAN_CelebA

First of all, you need to pre-train DCGAN on dataset CelebA-F, and Copy models G1 and D1 to **models**. 

We have saved the pre-trained models G1 and D1 that you can use directly, which can be downloaded from:
   https://github.com/EurekaTesla/ADT-GAN

Simply copy **G1** and **D1** in **Pre-trained_models/Pre-trained_models_CelebA** to **ADT-GAN/3.ADT-GAN/ADT-GAN_CelebA/models**.


## Preparation and configuation

# Assume that you have ccollect images and put them to any directory as mentioned in the README.md of the root directory

You may change the number of iteration  **Iteration** in **config.py**.
You may also set one or more absolute passes.
You may change directory path **Train_dirs** in **config.py**.
For example, 
```python
Train_dirs = [
    '/home/usrs/ADT-GAN/DATA/CelebA_64/CelebA-M',
]
```

### Usage

```bash
For training,
  $ python main.py --train
  
For testing,
   $ python main.py --test
```

The models are defined by **model.py**.

Generated images in training process are stored in **train_image** (defined in **config.py**).

Generated images used to calculate the FID are stored in **all_images** (defined in **config.py**).
