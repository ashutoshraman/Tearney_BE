import numpy as np
from PIL import Image
import cv2



def square_crop_img(img, scale):
    print(img.shape)
    mid_x, mid_y = img.shape[-1]/2, img.shape[-2]/2
    new_size_x, new_size_y = img.shape[-1]*scale, img.shape[-2]*scale
    left_x, right_x = mid_x - new_size_x/2, mid_x + new_size_x/2
    bottom_y, top_y = mid_y - new_size_y/2, mid_y + new_size_y/2
    try:
        new_img = img[:, round(left_x):round(right_x), round(bottom_y):round(top_y)]
    except IndexError:
        new_img = img[round(left_x):round(right_x), round(bottom_y):round(top_y)]
    return new_img

def downscale_img(image, x, y): #must be in (H, W, C) format, cannot be (C, H, W) so convert with transpose.(1, 2, 0)
    img_resized = cv2.resize(image, (0, 0), fx=x, fy=y, interpolation=cv2.INTER_LANCZOS4)
    return img_resized

def image_driver(path_data, path_masks, num, scale):
    img = Image.open(path_data + str(num) + '.tif').convert('RGB')
    msk = Image.open (path_masks + str(num) + '.png').convert('L')

    img.show()
    msk.show()

    
    np_img = np.array(img).transpose(2, 0, 1)
    np_msk = np.array(msk)

    new_img = square_crop_img(np_img, scale)
    new_msk = square_crop_img(np_msk, scale)


    new_image = Image.fromarray(new_img.astype('uint8').transpose(1, 2, 0), 'RGB')
    new_mask = Image.fromarray(new_msk.astype('uint8'), 'L')

    new_image.show()
    new_mask.show()

    ds_img = downscale_img(new_img.transpose(1, 2, 0), .5, .5)
    ds_msk = downscale_img(new_msk, .5, .5)

    ds_image = Image.fromarray(ds_img.astype('uint8'), 'RGB')
    ds_mask = Image.fromarray(ds_msk.astype('uint8'), 'L')

    ds_image.show()
    ds_mask.show()


    return



path_datas = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/images/"
path_mask = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/annotations/"

image_driver(path_datas, path_mask, 109, .85)


