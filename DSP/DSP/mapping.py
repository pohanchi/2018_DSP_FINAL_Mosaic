import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from scipy import misc

def map_loss(dot_org_img,x,y,size,feature_value):
    return np.sum(dot_org_img[x:x+size,y:y+size]) - feature_value
    
channel_map_list = pickle.load(open('feature_map_list.p','rb'))
final = np.zeros((20,30))
images_path = pickle.load(open('dataset.p','rb'))
list_path = 'imglist.p'
image_list=[]
if not os.path.exists(list_path):
    for num,i in enumerate(images_path):
        img=misc.imread(name = i,flatten = False , mode = 'RGB')
        tiny_image=misc.imresize(img,(64,64,3))
        
        image_list += [tiny_image]
    pickle.dump(image_list,open('imglist.p','wb'))
else:
    image_list = pickle.load(open('imglist.p','rb'))    
# for i in channel_map_list:
#     print(i)

org = pickle.load(open('feature_map_list.p','rb'))
org = np.array(org)
print(org.shape)
width = channel_map_list[0].shape[0]
height = channel_map_list[0].shape[1]
image = np.zeros((width,height))
#image[:]=1300

print(len(channel_map_list))

for i in range(0,width):
    for j in range(0,height):
        max_value = map_loss(org,i,j,64,channel_map_list[0][i][j])
        #max_value = channel_map_list[0][i][j]
        index = 0
        for k in range(0,len(channel_map_list)):
            if max_value < map_loss(org,i,j,64,channel_map_list[k][i][j]):
                max_value = map_loss(org,i,j,64,channel_map_list[k][i][j])
                index = k 
        image[i][j] = index

print(image)
final_image = np.zeros((1280,1920,3),dtype = 'uint8')
for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        #print(image_list[1].shape)
        #print(image[i,j])
        if int(image[i,j]) != 1300:
            final_image[i*64:(i+1)*64,j*64:(j+1)*64,:] = image_list[int(image[i,j])]
        else:
            final_image[i*64:(i+1)*64,j*64:(j+1)*64,:] = 0
plt.imshow(final_image)

final_save = np.zeros((1280,1920,3),dtype = 'uint8')
final_save[:,:,0] = final_image[:,:,2]
final_save[:,:,1] = final_image[:,:,0]
final_save[:,:,2] = final_image[:,:,1]
cv2.imwrite('final.jpg',final_save)
plt.show()
