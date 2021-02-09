##
# ADT-GAN program, the source code for paper "ADT-GAN: Adversarial and Discriminative Learning for Transferring Generative Adversarial" submittef for aaai21.
##

## Requirements
	tensorflow 1.X
	keras
	torch
	numpy
	matplotlib
	opencv-python
	tqdm

# Install the needed packages for running ADT-GAN program
```bash
When using CPU:
    $ pip install -r requirements.txt

When using GPU:
    $ pip install -r requirements_gpu.txt
```


## Function of each directory
1. **0.Pre-DCGAN**: You need to pre-train on dataset MNIST-not9 with Pre-DCGAN, and Copy models G1 and D1 to **models**. We have saved the pre-trained models G1 and D1 that you can use directly.

2. **1.DCGAN**: Training DCGAN on MNIST by **1.DCGAN/DCGAN_MNIST**. Training DCGAN on CelebA by **1.DCGAN/DCGAN_CelebA**.

3. **2.Initialized-DCGAN**: Training Initialized-DCGAN on MNIST by **2.Initialized-DCGAN/Initialized-DCGAN_MNIST**. Training Initialized-DCGAN on CelebA by **2.Initialized-DCGAN/Initialized-DCGAN_CelebA**.

4. **3.ADT-GAN**: Training ADT-GAN on MNIST by **3.ADT-GAN/ADT-GAN_MNIST**. Training ADT-GAN on CelebA by **3.ADT-GAN/ADT-GAN_CelebA**.

5. **FID**: The FID is the performance measure used to evaluate the experiments in the paper. 

## Please see the README.md under each of the above directory for guidance on running the program.

##

