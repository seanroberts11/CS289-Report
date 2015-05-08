import os
from PIL import Image
import numpy as np
from scipy import io

num_images = 61578

num_train = 50000
num_val = 5789
num_test = 5789

w = 424
h = 424
crop = 112
resize = 50


data_array = np.empty((num_images, resize * resize), dtype="uint8")

counter = 0
for im_file in sorted(os.listdir(os.getcwd() + "/images_training_rev1/")):
    if im_file.endswith(".jpg"):
        im = Image.open(os.getcwd() + "/images_training_rev1/" + im_file)
        imcrop = im.crop((crop, crop, w - crop, h - crop))
        imresize = imcrop.resize((resize,resize), Image.ANTIALIAS)
        imgrey = imresize.convert("L")
        data_array[counter] = np.array(imgrey, dtype="uint8").flatten()
        counter += 1
        if counter % 1000 == 0:
        	print(counter)

print(counter)


label_array = np.loadtxt(open("training_solutions_rev1.csv","rb"), delimiter=',', skiprows=1)[:,1:4]


rng = np.random.RandomState(85308)
initial_state = rng.get_state()
rng.shuffle(data_array)
rng.set_state(initial_state)
rng.shuffle(label_array)



galaxy_dict = {}
galaxy_dict['train_data'] = data_array[:num_train]
galaxy_dict['train_labels'] = label_array[:num_train]
galaxy_dict['val_data'] = data_array[num_train:num_train + num_val]
galaxy_dict['val_labels'] = label_array[num_train:num_train + num_val]
galaxy_dict['test_data'] = data_array[num_train + num_val:num_train + num_val + num_test]
galaxy_dict['test_labels'] = label_array[num_train + num_val:num_train + num_val + num_test]

io.savemat('galaxy_grey', galaxy_dict)
