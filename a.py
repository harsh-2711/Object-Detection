from pathlib import Path
from shutil import copyfile

f = open("training.csv","r")
lines = f.readlines()
counter = 0

for line in lines:
	img_name, x1,x2,y1,y2 = line.strip().split(',')
	file = Path(f'images/{img_name}')
	if file.is_file():
		counter += 1
		copyfile(f'images/{img_name}',f'train/{counter}.png')

f.close()
