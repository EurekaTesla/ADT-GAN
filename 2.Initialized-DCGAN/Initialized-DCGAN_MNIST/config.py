# -*- coding: utf-8 -*-

import os

## Network config
##  Input width and Height
Height, Width = 28, 28
Channel = 1

# input data shape
# channels_last -> [mb, c, h, w] , channels_first -> [mb, h, w, c]
Input_type = 'channels_last'

## Directory paths for training
Train_dirs = [
    #'/home/usrs/ADT-GAN/DATA/MNIST_PNG/MNIST-9'
]

Horizontal_flip = False
Vertical_flip = False
Rotate_ccw90 = False
Rotate_ccw180 = False


File_extensions = ['.jpg', '.png']

## Training config 
Iteration = 6000
Minibatch = 64

### Use or not history training (past generated images)
Use_history = False

## Test config
## The total number of generated images is Test_Minibatch * Test_num
Test_Minibatch = 64
Test_num = 100
Save_test_img_dir = 'test_images'

# if Save_combine is True, generated images in test are stored combined with same minibatch's
# if False, generated images are stored separately
# if None, generated image is not stored
Save_train_combine = True
Save_test_combine = False

Save_train_step = 50
Save_iteration_disp = True





#Generated images used to calculate the FID are saved every $x$ iterations.
Save_all_ima_step = 200
Save_all_img_dir = 'all_images/'






## Save config
Save_dir = 'models'
Save_d_name = 'D.h5'
Save_g_name = 'G.h5'
Save_d_path = os.path.join(Save_dir, Save_d_name)
Save_g_path = os.path.join(Save_dir, Save_g_name)
Save_train_img_dir = 'train_images'
Save_img_num = 5






## Other config
##  Randon_seed is used for seed of dataset shuffle in data_loader.py
Random_seed = 0

## Check
variety = ['channels_first', 'channels_last']
if not Input_type in variety:
    raise Exception("unvalid Input_type")

#os.system("rm {}/*".format(Save_train_img_dir))
import platform
python_version = int(platform.python_version_tuple()[0])
if python_version == 3:
    os.makedirs(Save_dir, exist_ok=True)
    os.makedirs(Save_train_img_dir, exist_ok=True)
    os.makedirs(Save_test_img_dir, exist_ok=True)
elif python_version == 2:
    if not os.path.exists(Save_dir):
        os.makedirs(Save_dir)
    if not os.path.exists(Save_train_img_dir):
        os.makedirs(Save_train_img_dir)
    if not os.path.exists(Save_test_img_dir):
        os.makedirs(Save_test_img_dir)
