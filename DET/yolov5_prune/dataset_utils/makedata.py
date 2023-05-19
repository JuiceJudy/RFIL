import shutil
import os
from tqdm import tqdm
 

# file_List = ["train2007", "val2007", "test2007"]
file_List = ["train2007", "val2007", "test2007","train2012", "val2012"]
for file in file_List:
    if not os.path.exists('G:/VOC/images/%s' % file):
        os.makedirs('G:/VOC/images/%s' % file)
    if not os.path.exists('G:/VOC/labels/%s' % file):
        os.makedirs('G:/VOC/labels/%s' % file)

    f = open('G:/voc/VOCdevkit/%s.txt' % file, 'r')
    lines = f.readlines()
    for line in tqdm(lines):
        line = line.strip()

        if os.path.exists(line):
            shutil.copy(line, "G:/VOC/images/%s" % file)
            line = line.replace('JPEGImages', 'labels')
            line = line.replace('jpg', 'txt')
            shutil.copy(line, "G:/VOC/labels/%s/" % file)
        else:
            print(line)