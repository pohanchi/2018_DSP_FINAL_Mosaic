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

def mosaic(img_path, tiles_path):
    print(tiles_path)
    tiles_data = TileProcessor(tiles_path).get_tiles()
    image_data,shape = TargetImage(img_path).get_data()
    preprocessing_time = time.time()
    print("It takes {} seconds on preprocessing.".format(preprocessing_time-start_time))
    compose(image_data, tiles_data,img_path,shape,img_path)

def search_max_value(width,height,list_):
    image=np.zeros(list_[0].shape)
    for i in range(0,width):
        for j in range(0,height):
            max_value = list_[0][i][j]
            index = 0
            for k in range(0,len(list_)):
                if max_value < list_[k][i][j]:
                    max_value = list_[k][i][j]
                    index = k 
            image[i][j] = index
    return image

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
    
    #define tensorflow operator -------------------------------------------------------------------------------------------------------#
    tmp_tiny = tf.placeholder(tf.float32, shape=(int(TILE_SIZE/TILE_BLOCK_SIZE) ,int(TILE_SIZE/TILE_BLOCK_SIZE) ,3,1))
    tmp_big  = tf.placeholder(tf.float32, shape=(1,int(w/TILE_BLOCK_SIZE),int(h/TILE_BLOCK_SIZE), 3))
    image_channel=tf.nn.conv2d(tmp_big,tmp_tiny,padding='VALID',strides=[1, TILE_SIZE/TILE_BLOCK_SIZE, TILE_SIZE/TILE_BLOCK_SIZE, 1])
    feature_map = tf.squeeze(image_channel)
    
    #run for first method--------------------------------------------------------------------------------------------------------------#
    feature_map_list =list()
    sess = tf.Session()
    hey=time.time()
    tt=np.asarray(tiles_small[0])
    print('small_picture.shape',tt.shape)

    for i in tiles_small:
        tmp=np.asarray(i)
        tmp=tmp[:,:,:,np.newaxis]
        tmp=tmp/255
        tmp=tmp*2-1
        feature_map_ = sess.run(feature_map,feed_dict={tmp_tiny:tmp,tmp_big:big_tmp})
        feature_map_list+= [feature_map_]
    ab=np.stack(feature_map_list)

    end=time.time()
    print('It take {} seconds'.format(end-hey))
    
    width = int(w/TILE_SIZE)
    height= int((h/TILE_SIZE))

    print('width {} height {}'.format(width,height))
    m_1 = time.time()
    # search index to fill in ---------------------------------------------------------------------------------------------------------#
    image=search_max_value(width,height,ab)

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
    
    



    
