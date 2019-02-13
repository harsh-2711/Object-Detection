from pathlib import Path
from shutil import copyfile

f = open("training.csv","r")
f_w = open("images.txt", "w")
lines = f.readlines()
counter = 0

for line in lines:
	if counter != 0:
		img_name, x1,x2,y1,y2 = line.strip().split(',')
		f_w.write(f'{counter} {img_name}\n')

	counter += 1

f.close()
f_w.close()