import pickle
import matplotlib
from scipy import misc

import numpy as np
import timeit
import PIL
from PIL import Image
import os
import time
import tensorflow as tf
import cv2 
import sys
import tqdm


TILE_SIZE      = 50		# height/width of mosaic tiles in pixels
TILE_MATCH_RES =  9 	# tile matching resolution (higher values give better fit but require more processing)
ENLARGEMENT    =  8		# the mosaic image will be this many times wider and taller than the original

TILE_BLOCK_SIZE = TILE_SIZE / max(min(TILE_MATCH_RES, TILE_SIZE), 1)
PIL.Image.MAX_IMAGE_PIXELS = 999287599

class TileProcessor():
    def __init__(self, tiles_directory):
        self.tiles_directory = tiles_directory
    def __process_tile(self, tile_path):
        try:
            img = Image.open(tile_path)
            # tiles must be square, so get the largest square that fits inside the image
            w = img.size[0]
            h = img.size[1]
            min_dimension = min(w, h)
            w_crop = (w - min_dimension) / 2
            h_crop = (h - min_dimension) / 2
            img = img.crop((w_crop, h_crop, w - w_crop, h - h_crop))
            large_tile_img = img.resize((TILE_SIZE, TILE_SIZE), Image.ANTIALIAS)
            small_tile_img = img.resize((int(TILE_SIZE/TILE_BLOCK_SIZE), int(TILE_SIZE/TILE_BLOCK_SIZE)), Image.ANTIALIAS)
            return (large_tile_img.convert('RGB'), small_tile_img.convert('RGB'))
        except:
            return (None, None)

    def get_tiles(self):
        large_tiles = []
        small_tiles = []

        print('Reading tiles from \'%s\'...' % (self.tiles_directory, ))
        # search the tiles directory recursively
        for root, subFolders, files in os.walk(self.tiles_directory):
            for tile_name in files:
                tile_path = os.path.join(root, tile_name)
                large_tile, small_tile = self.__process_tile(tile_path)
                if large_tile:
                    large_tiles.append(large_tile)
                    small_tiles.append(small_tile)
        
        print('Processed %s tiles.' % (len(large_tiles),))

        return (large_tiles, small_tiles)

class TargetImage():
    def __init__(self, image_path):
        self.image_path = image_path

    def get_data(self):
        print('Processing main image...')
        img = Image.open(self.image_path)
        w = img.size[0] * ENLARGEMENT
        h = img.size[1]	* ENLARGEMENT
        large_img = img.resize((w, h), Image.ANTIALIAS)
        w_diff = (w % TILE_SIZE)/2
        h_diff = (h % TILE_SIZE)/2
        # if necesary, crop the image slightly so we use a whole number of tiles horizontally and vertically
        if w_diff or h_diff:
            large_img = large_img.crop((w_diff, h_diff, w - w_diff, h - h_diff))
        T = np.asarray(large_img)
        small_img = large_img.resize((int(large_img.size[0]/TILE_BLOCK_SIZE), int(large_img.size[1]/TILE_BLOCK_SIZE)), Image.ANTIALIAS)
        print('small_img_shape',small_img.size)
        image_data = (large_img.convert('RGB'), small_img.convert('RGB'))

        print('Main image processed.')

        return image_data,T.shape

class MosaicImage:
    def __init__(self, original_img):
        self.image = Image.new(original_img.mode, original_img.size)
        self.x_tile_count = original_img.size[0] / TILE_SIZE
        self.y_tile_count = original_img.size[1] / TILE_SIZE
        self.total_tiles  = self.x_tile_count * self.y_tile_count
        return
    def add_tile(self, tile_data, coords):
        img = Image.new('RGB', (TILE_SIZE, TILE_SIZE))
        img.putdata(tile_data)
        self.image.paste(img, coords)
        return
    def save(self, path):
        self.image.save(path)
        return

def hsv_method_modify_vspace(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img[:,:,2][(img[:,:,2]/255>0.75) & (img[:,:,2]/255<0.9)] = (1.75-img[:,:,2][(img[:,:,2]/255>0.75) & (img[:,:,2]/255<0.9)])*img[:,:,2][(img[:,:,2]/255>0.75) & (img[:,:,2]/255<0.9)]
    img[:,:,2][(img[:,:,2]/255<0.25) & (img[:,:,2]/255>0.1)] = (1.25-img[:,:,2][(img[:,:,2]/255<0.25) & (img[:,:,2]/255>0.1)]/255)**2*img[:,:,2][(img[:,:,2]/255<0.25) & (img[:,:,2]/255>0.1)]
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def adjust_gamma(image, gamma=2.2):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def hsv_method_modify_Sspace(img):
    img = cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    img[:,:,1][(img[:,:,1]/255<0.15) & (img[:,:,1]/255>0.05)] = (1.15-img[:,:,1][(img[:,:,1]/255<0.15) & (img[:,:,1]/255>0.05)]/255)**2 * img[:,:,1][(img[:,:,1]/255<0.15) & (img[:,:,1]/255>0.05)]
    img = cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
    return img

def mosaic(img_path, tiles_path):
    print(tiles_path)
    tiles_data = TileProcessor(tiles_path).get_tiles()
    image_data,shape = TargetImage(img_path).get_data()
    preprocessing_time = time.time()
    print("It takes {} seconds on preprocessing.".format(preprocessing_time-start_time))
    compose(image_data, tiles_data,img_path,shape,img_path)


def fill_in_mosaic_map(shape,width,height,tiles_large,image):
    big_picture = np.zeros(shape)
    for i in tqdm.tqdm(range(width)):
        for j in range(height):
            big_picture[i*TILE_SIZE :(i+1)*TILE_SIZE, j*TILE_SIZE:(j+1)*TILE_SIZE,:] = np.asarray(tiles_large[int(image[i][j])])
    return big_picture

def image_save(name,big_picture):
    
    final_save = np.zeros(big_picture.shape,dtype = 'uint8')
    final_save[:,:,2] = big_picture[:,:,0]
    final_save[:,:,1] = big_picture[:,:,1]
    final_save[:,:,0] = big_picture[:,:,2]
    cv2.imwrite(name,final_save)
    
    return 

def map_loss_modify_for_division(feature_map_big_3,feature_map_tiny_3):
    loss_1 = sum(feature_map_big_3) / sum(feature_map_tiny_3) 
    loss_2_1 = feature_map_big_3[0] / (feature_map_tiny_3[0])
    loss_2_2 = feature_map_big_3[1] / (feature_map_tiny_3[1])
    loss_2_3 = feature_map_big_3[2] / (feature_map_tiny_3[2])
    return abs(1*loss_1 -1 + (loss_2_1 + loss_2_2 + loss_2_3 - 3)) 

def map_loss_item_1(feature_map_big_3,feature_map_tiny_3):
    loss_2_1 = (feature_map_big_3[0]) / (feature_map_tiny_3[0])
    loss_2_2 = (feature_map_big_3[1]) / (feature_map_tiny_3[1])
    loss_2_3 = (feature_map_big_3[2]) / (feature_map_tiny_3[2])
    return abs((loss_2_1 + loss_2_2 + loss_2_3 - 3))

def map_loss_item_2(feature_map_big,feature_map_tiny):
    loss_1 = (feature_map_big) / (feature_map_tiny) 
    return abs(loss_1-1)

def second_method_multiply_operators():
    tmp_2                    = tf.placeholder(tf.float32,shape=(int(TILE_SIZE/TILE_BLOCK_SIZE),int(TILE_SIZE/TILE_BLOCK_SIZE), 3))
    tmp_big2                 = tf.placeholder(tf.float32,shape=(int(w/TILE_BLOCK_SIZE),int(h/TILE_BLOCK_SIZE), 3))
    multiply_both_1          = tf.squeeze(tf.nn.conv2d(tf.expand_dims(tf.expand_dims(tmp_big2[:,:,0],2),0),tf.expand_dims(tf.expand_dims(tmp_2[:,:,0],2),-1),padding='VALID',strides=[1, TILE_SIZE/TILE_BLOCK_SIZE, TILE_SIZE/TILE_BLOCK_SIZE, 1]),0)
    multiply_both_2          = tf.squeeze(tf.nn.conv2d(tf.expand_dims(tf.expand_dims(tmp_big2[:,:,1],2),0),tf.expand_dims(tf.expand_dims(tmp_2[:,:,1],2),-1),padding='VALID',strides=[1, TILE_SIZE/TILE_BLOCK_SIZE, TILE_SIZE/TILE_BLOCK_SIZE, 1]),0)
    multiply_both_3          = tf.squeeze(tf.nn.conv2d(tf.expand_dims(tf.expand_dims(tmp_big2[:,:,2],2),0),tf.expand_dims(tf.expand_dims(tmp_2[:,:,2],2),-1),padding='VALID',strides=[1, TILE_SIZE/TILE_BLOCK_SIZE, TILE_SIZE/TILE_BLOCK_SIZE, 1]),0)
    multiply_both	         = tf.concat([multiply_both_1,multiply_both_2,multiply_both_3],2)

    big_img_square=tf.square(tmp_big2)
    big_img_tmp=tf.squeeze(big_img_square)
    one_channel_big=tf.reduce_sum(big_img_tmp,axis=2)

    tmp_add_3 = tf.placeholder(tf.float32, shape=(int(TILE_SIZE/TILE_BLOCK_SIZE), int(TILE_SIZE/TILE_BLOCK_SIZE),3))
    tmp_result_3 = tf.reduce_sum(tmp_add_3,[0,1])
    return multiply_both, big_img_tmp, one_channel_big, tmp_result_3

def search_division_closed_to_one(width,height,feature_map_big_3,feature_map_tiny_3_list):

    image = np.zeros((width,height))
    number = len(feature_map_tiny_3_list)
    time_ = time.time()
    for i in range(width):
        for j in range(height):
            min_value = map_loss_modify_for_division(feature_map_big_3[i,j],feature_map_tiny_3_list[0][i,j])
            index = 0
            for k in range(1,number):
                (min_value,index)=mini(min_value,index,k,feature_map_big_3[i,j],feature_map_tiny_3_list[k][i,j])
            image[i, j] = index
        if (i + 1) % 50 == 0:
            pass
    return image 


def mini(loss_min,index,current_index,feature_map_big_3,feature_map_tiny_3):
    loss_1 = map_loss_item_1(feature_map_big_3,feature_map_tiny_3)
    if loss_1 > loss_min:
        return (loss_min,index)
    loss_2 = map_loss_item_2(sum(feature_map_big_3),sum(feature_map_tiny_3))
    if loss_2 >loss_min:
        return (loss_min,index)
    loss_total=loss_1 + loss_2    
    if (loss_1 + loss_2) >= loss_min:
        return (loss_min, index)
    else:
        return(loss_total,current_index)

def compose(original_img, tiles,img_path,shape,name):

    print('Building mosaic, press Ctrl-C to abort...')
    original_img_large, original_img_small = original_img
    tiles_large, tiles_small = tiles

    mosaic = MosaicImage(original_img_large)
    w = shape[0]
    h = shape[1]
    big_tmp = np.asarray(original_img_small)
    print('big_tmp_shape {} {}'.format(big_tmp.shape[0],big_tmp.shape[1]))

    big_tmp[big_tmp==0] ==1
    big_tmp = big_tmp/255

    big_tmp = big_tmp * 2 - 1
    big_tmp_= big_tmp
    big_tmp = big_tmp[np.newaxis, :]
    
    
    #define second method multiply operators-------------------------------------------------------------------------------------------#
    #     
    
    tmp_2                    = tf.placeholder(tf.float32,shape=(int(TILE_SIZE/TILE_BLOCK_SIZE),int(TILE_SIZE/TILE_BLOCK_SIZE), 3))
    tmp_big2                 = tf.placeholder(tf.float32,shape=(int(w/TILE_BLOCK_SIZE),int(h/TILE_BLOCK_SIZE), 3))
    multiply_both_1          = tf.squeeze(tf.nn.conv2d(tf.expand_dims(tf.expand_dims(tmp_big2[:,:,0],2),0),tf.expand_dims(tf.expand_dims(tmp_2[:,:,0],2),-1),padding='VALID',strides=[1, TILE_SIZE/TILE_BLOCK_SIZE, TILE_SIZE/TILE_BLOCK_SIZE, 1]),0)
    multiply_both_2          = tf.squeeze(tf.nn.conv2d(tf.expand_dims(tf.expand_dims(tmp_big2[:,:,1],2),0),tf.expand_dims(tf.expand_dims(tmp_2[:,:,1],2),-1),padding='VALID',strides=[1, TILE_SIZE/TILE_BLOCK_SIZE, TILE_SIZE/TILE_BLOCK_SIZE, 1]),0)
    multiply_both_3          = tf.squeeze(tf.nn.conv2d(tf.expand_dims(tf.expand_dims(tmp_big2[:,:,2],2),0),tf.expand_dims(tf.expand_dims(tmp_2[:,:,2],2),-1),padding='VALID',strides=[1, TILE_SIZE/TILE_BLOCK_SIZE, TILE_SIZE/TILE_BLOCK_SIZE, 1]),0)
    multiply_both	         = tf.concat([multiply_both_1,multiply_both_2,multiply_both_3],2)

    big_img_square=tf.square(tmp_big2)
    big_img_tmp=tf.squeeze(big_img_square)
    one_channel_big=tf.reduce_sum(big_img_tmp,axis=2)

    tmp_add_3 = tf.placeholder(tf.float32, shape=(int(TILE_SIZE/TILE_BLOCK_SIZE), int(TILE_SIZE/TILE_BLOCK_SIZE),3))
    tmp_result_3 = tf.reduce_sum(tmp_add_3,[0,1])
    #----------------------------------------------------------------------------------------------------------------------------------#
    #run second method for division----------------------------------------------------------------------------------------------------#
    feature_map_three_list =list()
    sess = tf.Session()
    hey=time.time()
    tt=np.asarray(tiles_small[0])
    

    for i in tqdm.tqdm(range(len(tiles_small))):
        tmp=np.asarray(tiles_small[i])
        tmp[tmp==0] ==1
        tmp=tmp/255
        tmp = tmp * 2 - 1
        feature_map_three = sess.run(multiply_both, feed_dict={tmp_2: tmp, tmp_big2: big_tmp_})
        feature_map_three_list += [feature_map_three]

    big2_3 = sess.run(big_img_tmp,feed_dict={tmp_big2:big_tmp_})
    
    end=time.time()
    print('It take {} seconds'.format(end-hey))
    #define width, height for division method-----------------------------------------------------------------------------------------#
    width = int(w/TILE_SIZE)
    height= int((h/TILE_SIZE))

    print('width {} height {}'.format(width,height))

    feature_map_big_3	= np.zeros((width,height,3))
    #---------------------------------------------------------------------------------------------------------------------------------#
    #construct 3-channel feature map and one channel picture for big picture----------------------------------------------------------#
    for i in tqdm.tqdm(range(width)):
        for j in range(height):
            t_3=sess.run(tmp_result_3,feed_dict={tmp_add_3  :big2_3[i*int(TILE_SIZE/TILE_BLOCK_SIZE):(i+1)*int(TILE_SIZE/TILE_BLOCK_SIZE),j*int(TILE_SIZE/TILE_BLOCK_SIZE):(j+1)*int(TILE_SIZE/TILE_BLOCK_SIZE),:]})
            feature_map_big_3[i,j] = t_3
    m_1 = time.time()
    big_map_time =m_1 - end
    print('It calculate {} seconds on big_map'.format(big_map_time))
    #----------------------------------------------------------------------------------------------------------------------------------#
    # search index to fill in ---------------------------------------------------------------------------------------------------------#
    image=search_division_closed_to_one(width,height,feature_map_big_3,feature_map_three_list)
    
    m_3 = time.time()
    search_index_time = m_3 - m_1
    print('It calculate {} seconds on division step'.format(search_index_time))
    #----------------------------------------------------------------------------------------------------------------------------------#
    #fill in --------------------------------------------------------------------------------------------------------------------------#
    big_picture=fill_in_mosaic_map(shape,width,height,tiles_large,image)
    m_4 = time.time()
    time_fill_in = m_4-m_3
    print('It calculate {} seconds on fill-in step'.format(time_fill_in))
    #----------------------------------------------------------------------------------------------------------------------------------#
    #image save------------------------------------------------------------------------------------------------------------------------#
    image_save('final_fast_{}.jpg'.format(name[:-4]),big_picture)
    m_5=time.time()
    time_save=m_5-m_4
    print('It calculate {} seconds on save_fig step'.format(time_save))
    #----------------------------------------------------------------------------------------------------------------------------------#
    
    return


if __name__ == "__main__":
    start_time = time.time()

    if len(sys.argv) < 3:
        print('Usage: %s <image> <tiles directory>\r' % (sys.argv[0],))
    else:
        mosaic(sys.argv[1], sys.argv[2])
    # setup environment
    print('Building mosaic, press Ctrl-C to abort...')

    end_final_time = time.time()
    print('It take {} seconds on all time'.format(end_final_time-start_time))
    
    



    
