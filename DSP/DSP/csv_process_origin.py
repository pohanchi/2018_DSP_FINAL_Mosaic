import pickle
import matplotlib
from scipy import misc

import numpy as np
import timeit
from PIL import Image
import os
import time
import tensorflow as tf
import cv2 

if __name__ == "__main__":
    # setup environment
    start_time = time.time()
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction=0.95
    config.gpu_options.allow_growth = True

    #get_data
    original_image_path = 'test_0001.jpg'
    images_path = pickle.load(open('dataset.p','rb'))
    list_path = 'imglist_32.p'
    image_list=[]
    if not os.path.exists(list_path):
        for num,i in enumerate(images_path):
            img=misc.imread(name = i,flatten = False , mode = 'RGB')
            tiny_image=misc.imresize(img,(32,32,3))
            image_list += [tiny_image]
        pickle.dump(image_list,open('imglist_32.p','wb'))
    else:
        image_list = pickle.load(open('imglist_32.p','rb'))    
    big_img = misc.imread(name=original_image_path,flatten=False,mode='RGB')
    big_img = misc.imresize(big_img,(5120,7680,3))
    # big_img = cv2.cvtColor(big_img,cv2.COLOR_RGB2HSV)
    # big_img[:,:,2][(big_img[:,:,2]/255>0.85) & (big_img[:,:,2]/255<0.95) ] = np.floor(255*0.85)
    # big_img[:,:,2][(big_img[:,:,2]/255<0.25) & (big_img[:,:,2]/255>0.15) ] = np.ceil(255*0.25)
    # big_img = cv2.cvtColor(big_img,cv2.COLOR_HSV2RGB)
    big_img = big_img[np.newaxis,:]
    big_img = big_img/255
    big_img = big_img*2 - 1
    #define tensorflow operator
    tmp_tiny = tf.placeholder(tf.float32, shape=(32, 32,3,1))
    tmp_big  = tf.placeholder(tf.float32, shape=(1,5120, 7680,3))
    image_channel=tf.nn.conv2d(tmp_big,tmp_tiny,padding='VALID',strides=[1, 32, 32, 1])
    feature_map = tf.squeeze(image_channel)
   
    #run 
    channel_map_list = list()
    start_mid_time = time.time()
    print("It takes {} seconds on preprocessing.".format(start_mid_time-start_time))
    sess = tf.Session()
    if not os.path.exists('feature_map_list_32HSV.p'):
        for i in range(0,len(image_list)):
            tmp = image_list[i]
            # tmp = cv2.cvtColor(tmp,cv2.COLOR_RGB2HSV)
            # tmp[:,:,2][(tmp[:,:,2]/255>0.75) & (tmp[:,:,2]/255<0.9)] = np.floor(255*0.75)
            # tmp[:,:,2][(tmp[:,:,2]/255<0.25) & (tmp[:,:,2]/255<0.1)] = np.ceil(255*0.25)
            # tmp = cv2.cvtColor(tmp,cv2.COLOR_HSV2RGB)
            tmp=(tmp/255)*2-1
            tmp=tmp[:,:,:,np.newaxis]
            channel_map=sess.run(feature_map,feed_dict={tmp_tiny:tmp,tmp_big:big_img})
            channel_map_list += [channel_map]
            if (i+1) % 400 == 0 :
                print(i+1) 
        pickle.dump(channel_map_list,open('feature_map_list_32HSV.p','wb'))
    else:
        channel_map_list = pickle.load(open('feature_map_list_32HSV.p','rb'))
    width = channel_map_list[0].shape[0]
    height = channel_map_list[0].shape[1]
    print('width {} height {}'.format(width,height))
    image = np.zeros((width,height))
    end_mid_time = time.time()
    print('It take {} seconds on convolution operator'.format(end_mid_time-start_mid_time))
    print(len(channel_map_list))
    

    for i in range(0,width):
        for j in range(0,height):
            max_value = channel_map_list[0][i][j]
            index = 0
            for k in range(0,len(channel_map_list)):
                if max_value < channel_map_list[k][i][j]:
                    max_value = channel_map_list[k][i][j]
                    index = k 
            image[i][j] = index

    final_image = np.zeros((5120,7680,3),dtype = 'uint8')
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
                final_image[i*32:(i+1)*32,j*32:(j+1)*32,:] = image_list[int(image[i,j])]
    # plt.imshow(final_image)

    final_save = np.zeros((5120,7680,3),dtype = 'uint8')
    final_save[:,:,0] = final_image[:,:,2]
    final_save[:,:,1] = final_image[:,:,1]
    final_save[:,:,2] = final_image[:,:,0]
    cv2.imwrite('final_5.jpg',final_save)

    end_final_time = time.time()
    print('It take {} seconds on search time'.format(end_final_time-end_mid_time))

    # plt.show()

    end_time = time.time()
    print('It take {} seconds on all process'.format(end_time-start_time))



    
