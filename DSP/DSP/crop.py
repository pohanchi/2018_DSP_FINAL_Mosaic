import os 
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
import time
img_names=os.listdir('COMMON_image')
img_names_1 = os.listdir('METHOD1_image')
img_names_2 = os.listdir('METHOD2_image')
img_names_3 = os.listdir('METHOD3_image')
img_names_4 = os.listdir('METHOD4_image')
img_names_5 = os.listdir('origin_image')


for i in img_names_5:
    image=Image.open('origin_image/'+i)
    (w,h)=image.size
    Ratio = 4
    new_w = int(w/Ratio)
    new_h = int(h/Ratio)
    small = image.resize( (new_w, new_h), Image.BILINEAR )
    small_=ImageDraw.Draw(small)
    box = (0,h/2,w/2,h)
    small_.rectangle(((0, new_h/2), (new_w/2, new_h),), fill=None,outline='red',width=10)
    time.sleep(1.5)
    cropped = image.crop(box)
    box2 = (0,h/8,w/8,h/4)
    cropped_=ImageDraw.Draw(cropped)
    cropped_.rectangle(((0, h/8), (w/8, h/4)),fill=None,outline='red',width=10)
    time.sleep(1.5)
    cropped_2 = cropped.crop(box2)
    
    box3 = (0,h/16,w/16,h/8)
    cropped_2_=ImageDraw.Draw(cropped_2)
    cropped_2_.rectangle(((0, h/16), (w/16, h/8)),fill=None,outline='red',width=10)
    time.sleep(1.5)
    cropped_3  = cropped_2.crop(box3)

    small.save('origin_image/'+'small_'+i)
    cropped.save('origin_image/'+'small_crop_'+i)
    cropped_2.save('origin_image/'+'small_crop2_'+i)
    cropped_3.save('origin_image/'+'small_crop3_'+i)






