# Initialized-DCGAN_CelebA

Before that, you need to pre-train on dataset CelebA-F with DCGAN(0.Pre-DCGAN/DCGAN_CelebA), and Copy models **G1.h5** and **D1.h5** to **Initialized-DCGAN_CelebA/models**. 

We have saved the pre-trained models that you can use directly. Downloading Pre-trained_models **G1.h5** and **D1.h5** in **ADT-GAN/2.Initialized-DCGAN/Initialized-DCGAN_CelebA/models**.


## Preparation and configuation

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
When training,
$ python main.py --train

When testing,
$ python main.py --test
```

The models are defined by **model.py**.

Generated images in training process are stored in **train_image** (defined in **config.py**).

Generated images used to calculate the FID are stored in **all_images** (defined in **config.py**).
