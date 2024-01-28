import glob
import os
import numpy as np
import sys

current_dir = "data/Multiple_images"
current_dir1 = "Multiple_images"
file_train = open("train.txt", "w")
file_val = open("test.txt", "w")
counter = 1
pull_up_images= 118
pull_up_train_ind = round(pull_up_images*0.8, 0)
push_up_images = 115
push_up_train_ind = round(push_up_images*0.8,0)
squat_images = 183
squat_train_ind = round(squat_images*0.8,0)
situp_images = 128
crunch_train_ind = round(situp_images*0.8,0)
for pathAndFilename in glob.iglob(os.path.join(current_dir1, "*.png")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        if 'pushup' in title:
                if int(title.split("_")[1])> push_up_train_ind:
                        file_val.write(current_dir + "/" + title + '.png' + "\n")
                else:
                        file_train.write(current_dir + "/" + title + '.png' + "\n")
        if 'squat' in title:
                if int(title.split("_")[1])> squat_train_ind:
                        file_val.write(current_dir + "/" + title + '.png' + "\n")
                else:
                        file_train.write(current_dir + "/" + title + '.png' + "\n")
        if 'situp' in title:
                if int(title.split("_")[1])> crunch_train_ind:
                        file_val.write(current_dir + "/" + title + '.png' + "\n")
                else:
                        file_train.write(current_dir + "/" + title + '.png' + "\n")
        if 'pullup' in title:
                if int(title.split("_")[1])> crunch_train_ind:
                        file_val.write(current_dir + "/" + title + '.png' + "\n")
                else:
                        file_train.write(current_dir + "/" + title + '.png' + "\n")


        counter +=1

file_train.close()
file_val.close()
