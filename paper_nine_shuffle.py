import os, shutil
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import copy

class sight:
    def __init__(self, img, img_no, model):
        self.img = img
        self.img_no = img_no
        self.model = model

    def n_shuffle(self, div):
        if 150 % div != 0:
            div = 50
            print("divisions must be a factor of 150, div=50 assigned")

        img_y = self.img
        img_n = self.img_no
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

        self.img_list = img_list
        self.split_list = split_list
        self.pos_list = pos_list
        self.compressed_img = img_y

        return img_list, split_list, pos_list, img_y, img_n

    def model_prediction(self):
        test_set = [image.img_to_array(div_image) for div_image in self.img_list]
        test_arr = np.array(test_set)

        prediction = self.model.predict(test_arr)
        self.result = np.where(prediction == np.amax(prediction))

        print(prediction)

    def redborder(self):
        split_img = self.split_list[self.result[0][0]]
        position = self.pos_list[self.result[0][0]]

        width, height = split_img.size
        img_large = self.compressed_img

        vertical = Image.new('RGB', (2, height), (100, 0, 0))
        horizontal = Image.new('RGB', (width, 2), (100, 0, 0))

        split_img.paste(vertical, (0, 0, 2, height))
        split_img.paste(vertical, (width-2, 0, width, height))
        split_img.paste(horizontal, (0, 0, width, 2))
        split_img.paste(horizontal, (0, height-2, width, height))

        img_large.paste(split_img, position)

        return img_large

base_dir = '/Users/zslindsey/Desktop/MF/paper'
val_yes_dir = os.path.join(base_dir, 'validation/yes')
val_no_dir = os.path.join(base_dir, 'validation/no')

test_yes_dir = os.path.join(base_dir, 'test/yes')

fnames_yes = [os.path.join(val_yes_dir, fname) for
        fname in os.listdir(val_yes_dir)]

fnames_test_yes = [os.path.join(test_yes_dir, fname) for
        fname in os.listdir(test_yes_dir)]

fname_no = os.path.join(val_no_dir, os.listdir(val_no_dir)[0])

class_img_test = sight(Image.open(fnames_test_yes[8]), Image.open(fname_no),
                                    keras.models.load_model('/Users/zslindsey/Desktop/MF/paper/paper_1.h5'))

div_images, split_list, pos_list, og_image, img_no = class_img_test.n_shuffle(25)
class_img_test.model_prediction()

red_bordered_image = class_img_test.redborder()
red_bordered_image.show()

#img_a.show()
#class_img_test.img_list[0].show()
"""
test = [image.img_to_array(div_image) for div_image in div_images]
test_arr = np.array(test)

#images_y = [image.load_img(img, target_size=(150,150)) for img in fnames_test_yes]
#test_og = [image.img_to_array(original_image) for original_image in images_y]
#og_arr = np.array(test_og)

#img_test = redborder(img_a, fnames_yes[0])
#img_test.show()

model = keras.models.load_model('/Users/zslindsey/Desktop/MF/paper/paper_1.h5')
#print(model.predict(og_arr))

prediction = model.predict(test_arr)
print(prediction)

result = np.where(prediction == np.amax(prediction))
print(result[0][0])

img_test = redborder(split_list[result[0][0]], pos_list[result[0][0]], og_image)
#split_list[result[0][0]].show()
img_test.show()
"""
"""
for div_image in div_images:
    div_image.show()
"""
