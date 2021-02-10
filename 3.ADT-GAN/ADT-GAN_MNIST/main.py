# -*- coding: utf-8 -*-
#!/usr/bin/env python


# tensorflow 1.X

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import keras
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)


import argparse
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import sys
from tqdm import tqdm

import config as cf
from data_loader import DataLoader
from model import *

np.random.seed(cf.Random_seed)

Height, Width = cf.Height, cf.Width
Channel = cf.Channel

class Main_train():
    def __init__(self):
        pass

    def train(self):
        g_opt = keras.optimizers.Adam(lr=0.0005, beta_1=0.5)
        d_opt = keras.optimizers.Adam(lr=0.0005, beta_1=0.5)
        d1_opt = keras.optimizers.Adam(lr=0.001, beta_1=0.5)#, decay=0.0000001
        g1_opt = keras.optimizers.Adam(lr=0.001, beta_1=0.5)

        ## Load network model
        g = load_model('models/G1.h5')
        G1 = load_model('models/G1.h5')
        d = load_model('models/D1.h5')##
        d1 = D1_model(Height=Height, Width=Width, channel=Channel)



        d.trainable = True
        for layer in d.layers:
            layer.trainable = True
        d.compile(loss='binary_crossentropy', optimizer=d_opt)
        g.compile(loss='binary_crossentropy', optimizer=g_opt)

        d.trainable = False
        for layer in d.layers:
            layer.trainable = False
        c = Combined_model(g=g, d=d)
        c.compile(loss='binary_crossentropy', optimizer=g_opt)

        d1.trainable = True
        for layer in d1.layers:
            layer.trainable = True
        d1.compile(loss='binary_crossentropy', optimizer=d1_opt)
        g.compile(loss='binary_crossentropy', optimizer=g1_opt)

        d1.trainable = False
        for layer in d1.layers:
            layer.trainable = False
        c1 = Combined_model(g=g, d=d1)
        c1.compile(loss='binary_crossentropy', optimizer=g1_opt)




        ## Prepare Training data
        dl_train = DataLoader(phase='Train', shuffle=True)

        ## Start Train
        print('-- Training Start!!')
        if cf.Save_train_combine is None:
            print("generated image will not be stored")
        elif cf.Save_train_combine is True:
            print("generated image write combined >>", cf.Save_train_img_dir)
        elif cf.Save_train_combine is False:
            print("generated image write separately >>", cf.Save_train_img_dir)

        #保存每次迭代生成器和判别器的损失
        fname = os.path.join(cf.Save_dir, 'loss.txt')
        f = open(fname, 'w')
        f.write("Step,G_loss,D_loss{}".format(os.linesep))

        # Store prior generator's output 
        g_output_history = np.zeros((cf.Minibatch, Height, Width, Channel), dtype=np.float32)

        for ite in range(cf.Iteration):
            ite += 1
            # Discremenator training 
            y = dl_train.get_minibatch(shuffle=True)
            input_noise = np.random.uniform(-1, 1, size=(cf.Minibatch, 100))
            g_output = g.predict(input_noise, verbose=0)
            g_output_history = g_output
            X = np.concatenate((y, g_output_history))
            g_output1 = G1.predict(input_noise, verbose=0)
            g_output_history1 = g_output1

            X1 = np.concatenate((y, g_output_history1))
            X2 = np.concatenate((g_output_history, g_output_history1))
            Y = [1] * cf.Minibatch + [0] * cf.Minibatch

            d_loss = d.train_on_batch(X, Y)



            # Domain discriminator  training
            d1_loss = d1.train_on_batch(X1, Y)
            d1_loss = d1.train_on_batch(X2, Y)


            
            # Generator training
            input_noise = np.random.uniform(-1, 1, size=(cf.Minibatch, 100))
            g_loss = c.train_on_batch(input_noise, [1] * cf.Minibatch)     
            g1_loss = c1.train_on_batch(input_noise, [1] * cf.Minibatch)

            
            
            

            con = '|'
            if ite % cf.Save_train_step != 0:
                for i in range(ite % cf.Save_train_step):
                    con += '>'
                for i in range(cf.Save_train_step - ite % cf.Save_train_step):
                    con += ' '
            else:
                for i in range(cf.Save_train_step):
                    con += '>'
            con += '| '
            con += "Iteration:{}, L1: {:.9f}, L2: {:.9f}, L3: {:.9f}, L4: {:.9f} ".format(ite, d_loss, g_loss, d1_loss, g1_loss)
            sys.stdout.write("\r"+con)

            if ite % cf.Save_train_step == 0 or ite == 1:
                print()
                f.write("{},{},{},{},{}{}".format(ite, d_loss, g_loss, d1_loss, g1_loss, os.linesep))
                # save weights
                d.save_weights(cf.Save_d_path)
                g.save_weights(cf.Save_g_path)

                g_output = g.predict(input_noise, verbose=0)
                # save some samples
                if cf.Save_train_combine is True:
                    save_images(g_output, index=ite, dir_path=cf.Save_train_img_dir)
                elif cf.Save_train_combine is False:
                    save_images_separate(g_output, index=ite, dir_path=cf.Save_train_img_dir)





                # save all samples 用于后续计算FID
            if ite % cf.Save_all_ima_step == 0 or ite == 1:
                save_root = cf.Save_all_img_dir
		if not os.path.isdir(save_root):
		    os.mkdir(save_root)
		if not os.path.isdir(save_root + str(ite)):
		    os.mkdir(save_root + str(ite))

		#img_list = os.listdir(root)	
		pbar = tqdm(total=cf.Test_num) #进度条模块，美观直观，并且基本不影响程序效率，
		for i in range(cf.Test_num):
		    input_noise = np.random.uniform(-1, 1, size=(cf.Test_Minibatch, 100))
                    g_output = g.predict(input_noise, verbose=0)
                    save_images_separate(g_output, index=i, dir_path=save_root + str(ite))
                    pbar.update(1)  











        f.close()
        ## Save trained model
        d.save(cf.Save_d_path)
        g.save(cf.Save_g_path)
        print('Model saved -> ', cf.Save_d_path, cf.Save_g_path)


class Main_test():
    def __init__(self):
        pass

    def test(self):
        ## Load network model
        g = load_model('models/G.h5')

        print('-- Test start!!')
        if cf.Save_test_combine is None:
            print("generated image will not be stored")
        elif cf.Save_test_combine is True:
            print("generated image write combined >>", cf.Save_test_img_dir)
        elif cf.Save_test_combine is False:
            print("generated image write separately >>", cf.Save_test_img_dir)
        pbar = tqdm(total=cf.Test_num)

        for i in range(cf.Test_num):
            input_noise = np.random.uniform(-1, 1, size=(cf.Test_Minibatch, 100))
            g_output = g.predict(input_noise, verbose=0)

            if cf.Save_test_combine is True:
                save_images(g_output, index=i, dir_path=cf.Save_test_img_dir)
            elif cf.Save_test_combine is False:
                save_images_separate(g_output, index=i, dir_path=cf.Save_test_img_dir)
            pbar.update(1)



def save_images(imgs, index, dir_path):
    # Argment
    #  img_batch = np.array((batch, height, width, channel)) with value range [-1, 1]
    B, H, W, C = imgs.shape
    batch= imgs * 127.5 + 127.5
    batch = batch.astype(np.uint8)
    w_num = np.ceil(np.sqrt(B)).astype(np.int)
    h_num = int(np.ceil(B / w_num))
    out = np.zeros((h_num*H, w_num*W), dtype=np.uint8)
    for i in range(B):
        x = i % w_num
        y = i // w_num
        out[y*H:(y+1)*H, x*W:(x+1)*W] = batch[i, ..., 0]
    fname = str(index).zfill(len(str(cf.Iteration))) + '.jpg'
    save_path = os.path.join(dir_path, fname)

    if cf.Save_iteration_disp:
        plt.imshow(out, cmap='gray')
        plt.title("iteration: {}".format(index))
        plt.axis("off")
        plt.savefig(save_path)
    else:
        cv2.imwrite(save_path, out)


def save_images_separate(imgs, index, dir_path):
    # Argment
    #  img_batch = np.array((batch, height, width, channel)) with value range [-1, 1]
    B, H, W, C = imgs.shape
    batch= imgs * 127.5 + 127.5
    batch = batch.astype(np.uint8)
    for i in range(B):
        save_path = os.path.join(dir_path, '{}_{}.jpg'.format(index, i))
        cv2.imwrite(save_path, batch[i])


def arg_parse():
    parser = argparse.ArgumentParser(description='CNN implemented with Keras')
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--test', dest='test', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = arg_parse()

    if args.train:
        main = Main_train()
        main.train()
    if args.test:
        main = Main_test()
        main.test()

    if not (args.train or args.test):
        print("please select train or test flag")
        print("train: python main.py --train")
        print("test:  python main.py --test")
        print("both:  python main.py --train --test")
