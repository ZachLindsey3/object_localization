import os, shutil
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import copy

def redborder(img, position, img_large):
    width, height = img.size

    vertical = Image.new('RGB', (2, height), (100, 0, 0))
    horizontal = Image.new('RGB', (width, 2), (100, 0, 0))

    img.paste(vertical, (0, 0, 2, height))
    img.paste(vertical, (width-2, 0, width, height))
    img.paste(horizontal, (0, 0, width, 2))
    img.paste(horizontal, (0, height-2, width, height))

    img_large.paste(img, position)

    return img_large

def iterate(count, pixel_num, image_loc):
    x_1 = image_loc[0]
    y_1 = image_loc[1]
    x_2 = image_loc[2]
    y_2 = image_loc[3]

    x_diff = (x_2 - x_1) / 2
    x_1_start = x_1 - x_diff
    x_2_start = x_1 + x_diff
    y_diff = (y_2 - y_1) / 2
    y_1_start = y_1 - y_diff
    y_2_start = y_1 + y_diff

    i = int(count) // 11
    j = count % 11

    x_1_trans = (j*pixel_num) + x_1_start
    y_1_trans = (i*pixel_num) + y_1_start

    x_2_trans = (j*pixel_num) + x_2_start
    y_2_trans = (i*pixel_num) + y_2_start

    translation = (int(x_1_trans), int(y_1_trans), int(x_2_trans), int(y_2_trans))
    return (translation)


def n_shuffle(path_y, path_n, div=50):
    if 150 % div != 0:
        return 1
    else:
        img_y = Image.open(path_y)
        img_n = Image.open(path_n)
        step_num = int(150 / div)
        step_int = int(div)
        div_num = int(step_num*step_num)

        shrink = (150, 150)
        img_y = img_y.resize(shrink)
        img_n = img_n.resize(shrink)

        test_img = [copy.deepcopy(img_n) for i in range(div_num)]

        pos_list = []
        split_list = []
        img_list = []

        for i in range(step_num):
            for j in range(step_num):
                div_count = (3*i) + j
                pos = (i*step_int, j*step_int, (i*step_int)+step_int, (j*step_int)+step_int)
                split = img_y.crop(pos)
                test_img[div_count].paste(split, pos)

                pos_list.append(pos)
                split_list.append(split)
                img_list.append(test_img[div_count])

        return img_list, split_list, pos_list, img_y, img_n

def pinpoint(positive_pic, positive_loc, blank_pic, pixel_num=5):
    cut_section = positive_pic.crop(positive_loc)
    center = (50, 50, 100, 100)
    centered_positive = copy.deepcopy(blank_pic)
    centered_positive.paste(cut_section, center)

    #Iterage across new blank_pic center in devisions of 5 (0-50)
    step_num = 11
    div_num = int(step_num * step_num)
    test_copies = [copy.deepcopy(blank_pic) for i in range(div_num)]

    img_list = []
    loc_list = []

    for count in range(div_num):
        test_loc = iterate(count, pixel_num, center)
        test_cut = centered_positive.crop(test_loc)

        test_copies[count].paste(test_cut, test_loc)

        img_list.append(test_copies[count])
        loc_list.append(test_loc)

    return img_list, loc_list, 50

base_dir = '/Users/zslindsey/Desktop/MF/paper'
val_yes_dir = os.path.join(base_dir, 'validation/yes')
val_no_dir = os.path.join(base_dir, 'validation/no')

test_yes_dir = os.path.join(base_dir, 'test/yes')

fnames_yes = [os.path.join(val_yes_dir, fname) for
        fname in os.listdir(val_yes_dir)]

fnames_test_yes = [os.path.join(test_yes_dir, fname) for
        fname in os.listdir(test_yes_dir)]

fname_no = os.path.join(val_no_dir, os.listdir(val_no_dir)[0])

div_images, split_list, pos_list, og_image, img_no = n_shuffle(fnames_test_yes[9], fname_no, 50)

test = [image.img_to_array(div_image) for div_image in div_images]
test_arr = np.array(test)

##Apply model



model = keras.models.load_model('/Users/zslindsey/Desktop/MF/paper/paper_1.h5')
"""
prediction = model.predict(test_arr)
print(prediction)

result = np.where(prediction == np.amax(prediction))
print(result[0][0])

img_test = redborder(split_list[result[0][0]], pos_list[result[0][0]], og_image)

img_test.show()
"""

##Analyze results

#given "correct" picture positive_pic, pinpoint
positive_pic = div_images[0]
positive_loc = pos_list[0]
blank_pic = img_no

print(positive_loc)
positive_pic.show()
blank_pic.show()

pinpoint_img_list, pinpoint_loc_list, trans_factor = pinpoint(positive_pic, positive_loc, blank_pic, 5)

pinpoint_img = [image.img_to_array(pinpoint_image) for pinpoint_image in pinpoint_img_list]
pinpoint_array = np.array(pinpoint_img)
pinpoint_loc_array = np.array(pinpoint_loc_list)

pinpoint_prediction =  model.predict(pinpoint_array)
#print(pinpoint_prediction[0])

loc_array = np.where(pinpoint_prediction > 0.5)
loc_index = loc_array[0]
detection_array = pinpoint_loc_array[loc_index]
#print(detection_array[0])

detection_array_avg = detection_array.mean(axis=0)
detection_array_loc = detection_array_avg - trans_factor
print(detection_array_loc)
