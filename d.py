f1 = open("final_answers.txt","r")
f2 = open("test.csv","r")
f_w = open("final.csv","w")

lines1 = f1.readlines()
print(len(lines1))
lines2 = f2.readlines()

counter = 0
for line in lines2:
	img_name, x1, x2, y1, y2 = line.strip().split(',')
	f_w.write(f'{img_name},{x1},{x2},{y1},{y2}\n')
	break

for i in range(12815):
	img_name, x1, x2, y1, y2 = lines2[i+1].strip().split(',')
	x1_n, x2_n, y1_n, y2_n = lines1[i].strip().split(',')
	if i == 12814:
		f_w.write(f'{img_name},{x1_n},{x2_n},{y1_n},{y2_n}')
	else:
		f_w.write(f'{img_name},{x1_n},{x2_n},{y1_n},{y2_n}\n')

f1.close()
f2.close()
f_w.close()
