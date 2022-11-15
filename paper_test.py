import os, shutil
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.keras.preprocessing import image
from tensorflow import keras

base_dir = '/Users/zslindsey/Desktop/MF/paper'

test_yes_dir = os.path.join(base_dir, 'test/yes')
test_no_dir = os.path.join(base_dir, 'test/no')

fnames_y = [os.path.join(test_yes_dir, fname) for
        fname in os.listdir(test_yes_dir)]
fnames_n = [os.path.join(test_no_dir, fname) for
        fname in os.listdir(test_no_dir)]

img_path_y = fnames_y[5:8]
img_path_n = fnames_n[5:8]

images_y = [image.load_img(img, target_size=(150,150)) for
        img in img_path_y]
images_n = [image.load_img(img, target_size=(150,150)) for
        img in img_path_n]

y = [image.img_to_array(img) for
        img in images_y]
n = [image.img_to_array(img) for
        img in images_n]

yy = np.array(y)
nn = np.array(n)
print(yy.shape)
print(nn.shape)
#y = x.reshape((1,) + x.shape)

model = keras.models.load_model('/Users/zslindsey/Desktop/MF/paper/paper_1.h5')
print(model.summary())

#tom_dir = os.path.join(base_dir, 'test/tom.jpg')
#tom_img = image.load_img(tom_dir, target_size=(150,150))

#tom_arr = image.img_to_array(tom_img)
#tom_test = tom_arr.reshape((1,) + tom_arr.shape)

#print(model.predict(tom_test))
"""
plt.figure(0)
imgplot = plt.imshow(image.array_to_img(tom_arr))
plt.show()
"""
print("yes:", sep='\n')
print(model.predict(yy))
print("no:", sep='\n')
print(model.predict(nn))

i=0
for img in y:
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(img))

    i+=1

for img in n:
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(img))

    i+=1

plt.show()
