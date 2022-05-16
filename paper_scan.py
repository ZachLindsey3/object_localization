import os, shutil
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow import keras
import copy
import matplotlib.pyplot as plt

def sectional_expected_value(pdf_h, pdf_v, height_cutoff=0.5):
    #Note#
    #probablity is not normalized to 1
    #p=1 does not indicate 100% probablity, just that model thinks it is 100%

    #pdf_h is the horizontal axis pdf
    #pdf_v is the vertical axis pdf

    #identify index values with p>0.5
    loc_h = np.where(pdf_h > height_cutoff)[0]
    loc_v = np.where(pdf_v > height_cutoff)[0]

    #group adjacent index values into "areas of interest"
    interest_split_h = []
    interest_split_v = []
    for i in range(len(np.diff(loc_h))):
        if np.diff(loc_h)[i] > 1:
            h = i+1
            interest_split_h.append(h)

    for i in range(len(np.diff(loc_v))):
        if np.diff(loc_v)[i] > 1:
            v = i+1
            interest_split_v.append(v)

    prob_test_h = np.split(loc_h, np.cumsum(interest_split_h))
    prob_test_v = np.split(loc_v, np.cumsum(interest_split_v))

    #find areas of interest with the hightest total p, "expected range"
    roi_weight_h = np.asarray([np.sum(pdf_h[i]) for i in prob_test_h])
    expected_range_h = prob_test_h[(np.where(roi_weight_h == np.amax(roi_weight_h))[0])[0]]

    roi_weight_v = np.asarray([np.sum(pdf_v[i]) for i in prob_test_v])
    expected_range_v = prob_test_v[(np.where(roi_weight_v == np.amax(roi_weight_v))[0])[0]]

    #find the expected value within the expected range
    expected_location_h = np.dot(expected_range_h, pdf_h[expected_range_h]) / np.shape(expected_range_h)[0]
    expected_location_v = np.dot(expected_range_v, pdf_v[expected_range_v]) / np.shape(expected_range_v)[0]

    #return expected location as numpy array with dimension (1,)
    return(expected_location_h, expected_location_v)

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

    def redborder_adjustable(self, image, unit_size, x_loc, y_loc):
        x_coord = x_loc*int(unit_size/2)
        y_coord = y_loc*int(unit_size/2)
        position = (x_coord, y_coord, x_coord+unit_size, y_coord+unit_size)
        split_img = image.crop(position)

        width, height = split_img.size
        img_large = image

        vertical = Image.new('RGB', (2, height), (100, 0, 0))
        horizontal = Image.new('RGB', (width, 2), (100, 0, 0))

        split_img.paste(vertical, (0, 0, 2, height))
        split_img.paste(vertical, (width-2, 0, width, height))
        split_img.paste(horizontal, (0, 0, width, 2))
        split_img.paste(horizontal, (0, height-2, width, height))

        img_large.paste(split_img, position)

        return(img_large)

    def scan(self, bar_width):
        if 150 % bar_width != 0:
            bar_width = 10
            print("bar width must be a factor of 150, bar width=10 assigned")

        if bar_width % 2 != 0:
            bar_width = bar_width*2
            print("bar width must be an even number, bar width doubled")

        img_y = self.img
        #img_y.show()

        width, height = 150, 150
        shrink = (width, height)
        img_y = img_y.resize(shrink)
        h_scan = int(2*width / bar_width)
        v_scan = int(2*height / bar_width)

        bar_h_list = [copy.deepcopy(img_y) for i in range(h_scan)]
        bar_v_list = [copy.deepcopy(img_y) for i in range(v_scan)]
        h_bar = Image.new('RGB', (bar_width, height), (205, 133, 45))
        v_bar = Image.new('RGB', (width, bar_width), (205, 133, 45))

        for i in range(h_scan):
            bar_start_x = i*int(bar_width/2)
            bar_stop_x = bar_width + (i*int(bar_width/2))
            bar_h_list[i].paste(h_bar, (bar_start_x, 0, bar_stop_x, height))
            #bar_img_list[i].show()

        for i in range(v_scan):
            bar_start_y = i*int(bar_width/2)
            bar_stop_y = bar_width + (i*int(bar_width/2))
            bar_v_list[i].paste(v_bar, (0, bar_start_y, width, bar_stop_y))
            #bar_v_list[i].show()

        test_set_h = [image.img_to_array(bar_images) for bar_images in bar_h_list]
        test_arr_h = np.array(test_set_h)

        test_set_v = [image.img_to_array(bar_images) for bar_images in bar_v_list]
        test_arr_v = np.array(test_set_v)

        prediction_h = self.model.predict(test_arr_h)
        prediction_v = self.model.predict(test_arr_v)

        pdf_h = (-1*prediction_h) + 1
        pdf_v = (-1*prediction_v) + 1


        """
        #print(pdf_h)
        #print(pdf_v)

        #print(np.diff(pdf_h, axis=0))

        loc_h = np.where(pdf_h > 0.5)[0]
        loc_v = np.where(pdf_v > 0.5)[0]

        print(loc_h, loc_v)
        print(np.diff(loc_h), np.diff(loc_v))

        interest_split_h = []
        interest_split_v = []
        for i in range(len(np.diff(loc_h))):
            if np.diff(loc_h)[i] > 1:
                h = i+1
                interest_split_h.append(h)

        for i in range(len(np.diff(loc_v))):
            if np.diff(loc_v)[i] > 1:
                v = i+1
                interest_split_v.append(v)

        print(interest_split_h)
        prob_test_h = np.split(loc_h, np.cumsum(interest_split_h))
        print(prob_test_h)
        roi_weight_h = np.asarray([np.sum(pdf_h[i]) for i in prob_test_h])
        expected_range_h = prob_test_h[(np.where(roi_weight_h == np.amax(roi_weight_h))[0])[0]]
        expected_location_h = np.dot(expected_range_h, pdf_h[expected_range_h]) / np.shape(expected_range_h)[0]
        print(round(expected_location_h.item()))

        prob_test_v = np.split(loc_v, np.cumsum(interest_split_v))

        roi_weight_v = np.asarray([np.sum(pdf_v[i]) for i in prob_test_v])
        expected_range_v = prob_test_v[(np.where(roi_weight_v == np.amax(roi_weight_v))[0])[0]]
        expected_location_v = np.dot(expected_range_v, pdf_v[expected_range_v]) / np.shape(expected_range_v)[0]
        print(round(expected_location_v.item()))
        print(pdf_v)

        #bar_h_list[loc_h[0]].show()
        #bar_v_list[loc_v[0]].show()
        """
        horizontal_location, vertical_location = sectional_expected_value(pdf_h, pdf_v, height_cutoff=0.5)

        located_image = class_img_test.redborder_adjustable(img_y, bar_width,
                                                            round(horizontal_location.item()),
                                                            round(vertical_location.item()))
        located_image.show()

base_dir = '/Users/zslindsey/Desktop/MF/paper'
val_yes_dir = os.path.join(base_dir, 'validation/yes')
val_no_dir = os.path.join(base_dir, 'validation/no')

test_yes_dir = os.path.join(base_dir, 'test/yes')

fnames_yes = [os.path.join(val_yes_dir, fname) for
        fname in os.listdir(val_yes_dir)]

fnames_test_yes = [os.path.join(test_yes_dir, fname) for
        fname in os.listdir(test_yes_dir)]

fname_no = os.path.join(val_no_dir, os.listdir(val_no_dir)[0])


class_img_test = sight(Image.open(fnames_test_yes[9]), Image.open(fname_no),
                                keras.models.load_model('/Users/zslindsey/Desktop/MF/paper/paper_1.h5'))

class_img_test.scan(30)

"""
for scan_image in fnames_test_yes:
    class_img_test = sight(Image.open(scan_image), Image.open(fname_no),
                                    keras.models.load_model('/Users/zslindsey/Desktop/MF/paper/paper_1.h5'))

    class_img_test.scan(30)
"""

"""Notes:
bar_width 10, end to end, gives 5 full accuracy + 2 half accuracy
bar_width 10, half shift per frame, gives 5 full accuracy + 3 half accuracy
test 0 and 5
"""
