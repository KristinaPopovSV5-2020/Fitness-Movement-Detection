import glob
import os
import numpy as np
import sys

current_dir1 = "yolodemo/Multiple_images"
current_dir = "data/Multiple_images"
file_train = open("train1.txt", "w")
file_val = open("test1.txt", "w")
counter = 1
star_jump_images= 369
star_jump_train_ind = round(star_jump_images*0.8, 0)
push_up_images = 428
push_up_train_ind = round(push_up_images*0.8,0)
squat_images = 837
squat_train_ind = round(squat_images*0.8,0)
crunch_images = 182
crunch_train_ind = round(crunch_images*0.8,0)
for pathAndFilename in glob.iglob(os.path.join(current_dir1, "*.jpeg")):
        print(pathAndFilename)
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        print(title)
        if len(title) < 4:
                if int(title) > star_jump_train_ind:
                        file_val.write(current_dir + "/" + title + '.jpeg' + "\n")
                else:
                        file_train.write(current_dir + "/" + title + '.jpeg' + "\n")


        """if 'pushup' in title:
                if int(title.split("_")[1])> push_up_train_ind:
                        file_val.write(current_dir + "/" + title + '.png' + "\n")
                else:
                        file_train.write(current_dir + "/" + title + '.png' + "\n")
        if 'squat' in title:
                if int(title.split("_")[1])> squat_train_ind:
                        file_val.write(current_dir + "/" + title + '.png' + "\n")
                else:
                        file_train.write(current_dir + "/" + title + '.png' + "\n")
        if 'crunch' in title:
                if int(title.split("_")[1])> crunch_train_ind:
                        file_val.write(current_dir + "/" + title + '.png' + "\n")
                else:
                        file_train.write(current_dir + "/" + title + '.png' + "\n")
"""

        counter +=1

file_train.close()
file_val.close()
