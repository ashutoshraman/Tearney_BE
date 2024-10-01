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

def split_pic_in2(img, num_splits_hor, num_splits_vert): # do it in H, W, C for preservation of order
    height, width = img.shape[0], img.shape[1]
    print(f"height is {height}, width is {width}")
    height_cutoff, width_cutoff = height // num_splits_vert, width // num_splits_hor
    s1 = img[:height_cutoff, :width_cutoff]
    s2 = img[height_cutoff:, width_cutoff:]
    return s1, s2

def split_pic(img, hor_split, vert_split): # in H, W, C format
    # tiles = []
    # for y in np.linspace(0, img.shape[0], vert_split+1):
    #     for x in np.linspace(0, img.shape[1], hor_split+1):
    #         tile = img[x:x+(img.shape[1]//hor_split), y:y+(img.shape[0]//vert_split)]

    # tiles = [img[x:x+(img.shape[0]//hor_split), y:y+(img.shape[1]//vert_split)]  ]
    # return tiles
    H, W = img.shape[0], img.shape[1]
    M, N = H//hor_split, W// vert_split
    tiles = [img[x:x+M,y:y+N] for x in range(0, H-M, M) for y in range(0, W-M, N)]
    return tiles


def fold_batch(img, fold_ratio):
    pass

def image_driver(path_data, path_masks, num, scale, fx, fy):
    img = Image.open(path_data + str(num) + '.tif').convert('RGB')
    msk = Image.open (path_masks + str(num) + '.png').convert('L')

    # img.show()
    # msk.show()

    
    np_img = np.array(img).transpose(2, 0, 1)
    np_msk = np.array(msk)

    new_img = square_crop_img(np_img, scale)
    new_msk = square_crop_img(np_msk, scale)


    new_image = Image.fromarray(new_img.astype('uint8').transpose(1, 2, 0), 'RGB')
    new_mask = Image.fromarray(new_msk.astype('uint8'), 'L')

    # new_image.show()
    # new_mask.show()

    ds_img = downscale_img(new_img.transpose(1, 2, 0), fx, fy) # convert to H, W, C
    ds_msk = downscale_img(new_msk, fx, fy)

    ds_image = Image.fromarray(ds_img.astype('uint8'), 'RGB')
    ds_mask = Image.fromarray(ds_msk.astype('uint8'), 'L')

    # ds_image.show()
    # ds_mask.show()

    split_imgs = split_pic(ds_img, 2, 2)
    split_msks = split_pic(ds_msk, 2, 2)

    for i in range(len(split_imgs)):
        Image.fromarray(split_imgs[i].astype('uint8'), 'RGB').show()
        Image.fromarray(split_msks[i].astype('uint8'), 'L').show()

   
    


    return



path_datas = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/images/"
path_mask = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/annotations/"

image_driver(path_datas, path_mask, 109, .85, .5, .5)


