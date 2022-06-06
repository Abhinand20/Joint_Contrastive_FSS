import os

img_path='data/images/val'
txt_path = 'val.txt'
fw = open(txt_path, "w")

for filename in os.listdir(img_path):
    fw.write(filename.split('.')[0] + '\n')

fw.close()
