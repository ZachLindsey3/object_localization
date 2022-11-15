import os, shutil
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import copy

def redborder(img, position, img_large):
    width, height = img.size

    vertical = Image.new('RGB', (4, height), (100, 0, 0))
    horizontal = Image.new('RGB', (width, 4), (100, 0, 0))

    img.paste(vertical, (0, 0, 4, height))
    img.paste(vertical, (width-4, 0, width, height))
    img.paste(horizontal, (0, 0, width, 4))
    img.paste(horizontal, (0, height-4, width, height))

    img_large.paste(img, position)

    return img_large

def half_shuffle(path_y, path_n):
    img_1 = Image.open(path_y)
    img_2 = Image.open(path_n)

    shrink = (150, 150)
    img_1 = img_1.resize(shrink)
    img_2 = img_2.resize(shrink)

    left_test = copy.deepcopy(img_2)
    right_test = copy.deepcopy(img_2)

    left = (0, 0, 75, 150)
    right = (75, 0, 150, 150)
    split_l = img_1.crop(left)
    split_r = img_1.crop(right)

    split = [split_l, split_r]
    side = [left, right]

    left_test.paste(split_l, left)
    right_test.paste(split_r, right)

    return left_test, right_test, split, side, img_1

base_dir = '/Users/zslindsey/Desktop/MF/paper'
val_yes_dir = os.path.join(base_dir, 'validation/yes')
val_no_dir = os.path.join(base_dir, 'validation/no')

fnames_yes = [os.path.join(val_yes_dir, fname) for
        fname in os.listdir(val_yes_dir)]

fname_no = os.path.join(val_no_dir, os.listdir(val_no_dir)[0])

img_a, img_b, split_mat, side_mat, og_image = half_shuffle(fnames_yes[3], fname_no)

#img_a.show()
#img_b.show()

test = [image.img_to_array(img_a), image.img_to_array(img_b)]
test_arr = np.array(test)

#img_test = redborder(img_a, fnames_yes[0])
#img_test.show()

model = keras.models.load_model('/Users/zslindsey/Desktop/MF/paper/paper_1.h5')

result = model.predict(test_arr)
#print(result)

prob_left = result[0,0]
prob_right = result[1,0]
if prob_left > prob_right:
    print("object is on the left with probability " + str(prob_left))
    img_test = redborder(split_mat[0], side_mat[0], og_image)
else:
    print("object is on the right with probablilty " + str(prob_right))
    img_test = redborder(split_mat[1], side_mat[1], og_image)

img_test.show()
