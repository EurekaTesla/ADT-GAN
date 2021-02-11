# Pre-DCGAN_CelebA

## Preparation and configuation

You may change the number of iteration  **Iteration** in **config.py**.
You may also set one or more absolute passes.
You may change directory path **Train_dirs** in **config.py**.
For example, 
```python
Train_dirs = [
    '/home/usrs/ADT-GAN/DATA/CelebA_64/CelebA-F'
]
```

### Usage

```bash
When training,
$ python main.py --train
```

The models are defined by **model.py**
