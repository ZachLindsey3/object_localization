import os, shutil

import whatimage
import pyheif
from PIL import Image
from tensorflow.keras.preprocessing import image

base_dir = '/Users/zslindsey/Desktop/MF/paper'

yes_dir = os.path.join(base_dir, 'yes_paper')
no_dir = os.path.join(base_dir, 'no_paper')

train_dir = os.path.join(base_dir, 'train')
#os.mkdir(train_dir)
validation_dir = os.path.join(base_dir, 'validation')
#os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
#os.mkdir(test_dir)

train_yes_dir = os.path.join(train_dir, 'yes')
#os.mkdir(train_yes_dir)
train_no_dir = os.path.join(train_dir, 'no')
#os.mkdir(train_no_dir)

val_yes_dir = os.path.join(validation_dir, 'yes')
#os.mkdir(val_yes_dir)
val_no_dir = os.path.join(validation_dir, 'no')
#os.mkdir(val_no_dir)

test_yes_dir = os.path.join(test_dir, 'yes')
#os.mkdir(test_yes_dir)
test_no_dir = os.path.join(test_dir, 'no')
#os.mkdir(test_no_dir)

fnames_yes = [os.path.join(yes_dir, fname) for
        fname in os.listdir(yes_dir)]
fnames_no = [os.path.join(no_dir, fname) for
        fname in os.listdir(no_dir)]

ds_y = os.path.join(yes_dir, '.DS_Store')
ds_n = os.path.join(no_dir, '.DS_Store')

fnames_yes.remove(ds_y)
fnames_no.remove(ds_n)

#print(len(fnames_no))
#print(len(fnames_yes))

i=0
while i<70:
    img_y = pyheif.read(fnames_yes[i])
    img_n = pyheif.read(fnames_no[i])

    img_conv_y = Image.frombytes(
            img_y.mode,
            img_y.size,
            img_y.data,
            "raw",
            img_y.mode,
            img_y.stride,
            )
    img_conv_n = Image.frombytes(
            img_n.mode,
            img_n.size,
            img_n.data,
            "raw",
            img_n.mode,
            img_n.stride,
            )

    file_name_y = "yes_"+str(i).zfill(2)+".jpg"
    file_dir_y = os.path.join(train_yes_dir, file_name_y)
    file_name_n = "no_"+str(i).zfill(2)+".jpg"
    file_dir_n = os.path.join(train_no_dir, file_name_n)

    img_conv_y.save(file_dir_y, "JPEG")
    img_conv_n.save(file_dir_n, "JPEG")
    i+=1

while i<90:
    img_y = pyheif.read(fnames_yes[i])
    img_n = pyheif.read(fnames_no[i])

    img_conv_y = Image.frombytes(
            img_y.mode,
            img_y.size,
            img_y.data,
            "raw",
            img_y.mode,
            img_y.stride,
            )
    img_conv_n = Image.frombytes(
            img_n.mode,
            img_n.size,
            img_n.data,
            "raw",
            img_n.mode,
            img_n.stride,
            )

    file_name_y = "yes_"+str(i).zfill(2)+".jpg"
    file_dir_y = os.path.join(val_yes_dir, file_name_y)
    file_name_n = "no_"+str(i).zfill(2)+".jpg"
    file_dir_n = os.path.join(val_no_dir, file_name_n)

    img_conv_y.save(file_dir_y, "JPEG")
    img_conv_n.save(file_dir_n, "JPEG")
    i+=1

while i<100:
    img_y = pyheif.read(fnames_yes[i])
    img_n = pyheif.read(fnames_no[i])

    img_conv_y = Image.frombytes(
            img_y.mode,
            img_y.size,
            img_y.data,
            "raw",
            img_y.mode,
            img_y.stride,
            )
    img_conv_n = Image.frombytes(
            img_n.mode,
            img_n.size,
            img_n.data,
            "raw",
            img_n.mode,
            img_n.stride,
            )

    file_name_y = "yes_"+str(i).zfill(2)+".jpg"
    file_dir_y = os.path.join(test_yes_dir, file_name_y)
    file_name_n = "no_"+str(i).zfill(2)+".jpg"
    file_dir_n = os.path.join(test_no_dir, file_name_n)

    img_conv_y.save(file_dir_y, "JPEG")
    img_conv_n.save(file_dir_n, "JPEG")
    i+=1

print('total training yes images:', len(os.listdir(train_yes_dir)))
print('total training no images:', len(os.listdir(train_no_dir)))
print('total validation yes images:', len(os.listdir(val_yes_dir)))
print('total validation no images:', len(os.listdir(val_no_dir)))
print('total test yes images:', len(os.listdir(test_yes_dir)))
print('total test no images:', len(os.listdir(test_no_dir)))
