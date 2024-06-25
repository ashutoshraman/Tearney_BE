import os, sys
import PIL
from PIL import Image 
import numpy as np
import pickle
# plan is open all images in for loop of directory, check for grayscale
# color value of 76 to confirm BE and say label all as healthy but if
# 76 is found even once then relabel as BE, then create dict with filename
# or number and corresponding label, either save dict or run in be_segment.py


path_masks = "/Users/ashutoshraman/Documents/repos/Tearney_BE/raw_data/annotations/"

masks_list = []
keys_list = []
for i in sorted(os.listdir(path_masks)):
    new_i = i.strip('.png')
    try:
        img = Image.open(os.path.join(path_masks, i)).convert("L")
        img_array = np.array(img, dtype=np.int32)
        if 76 in img_array:
            label= 1
        else:
            label= 0
        masks_list.append(label)
        keys_list.append(new_i)
    except PIL.UnidentifiedImageError:
        pass
label_dict = dict(zip(keys_list, masks_list))

   
with open("labels.pkl", 'wb') as filename:
    pickle.dump(label_dict, filename)
    print("dictionary saved successfully")


# with open("labels.pkl", 'rb') as new_dict:
#     final_dict = pickle.load(new_dict)
#     print(final_dict)

