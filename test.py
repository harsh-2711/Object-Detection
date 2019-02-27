import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import load_model
import random
from PIL import Image
import numpy as np
import pickle

plt.switch_backend('agg')


def Normalize(image,mean,std):
    for channel in range(3):
        image[:,:,channel]=(image[:,:,channel]-mean[channel])/std[channel]
    return image

id_to_data={}
id_to_size={}

with open("images.txt") as f:
    lines=f.read().splitlines()
    for line in lines:
        id,path=line.split(" ",1)
        print(id)
        image=Image.open("train/0/"+id+".png").convert('RGB')
        id_to_size[int(id)]=np.array(image,dtype=np.float32).shape[0:2]
        image=image.resize((224,224))
        image=np.array(image,dtype=np.float32)
        image=image/255
        image=Normalize(image,[0.485,0.456,0.406],[0.229,0.224,0.225])
        id_to_data[int(id)]=image

id_to_data=np.array(list(id_to_data.values()))
id_to_size=np.array(list(id_to_size.values()))

id_to_box={}

with open("bounding_boxes.txt") as f:
    lines=f.read().splitlines()
    for line in lines:
        id, box = line.split(" ",1)
        box = np.array([float(i) for i in box.split(" ")],dtype=np.float32)
        box[0]=box[0]/id_to_size[int(id)-1][1]*224
        box[1]=box[1]/id_to_size[int(id)-1][0]*224
        box[2]=box[2]/id_to_size[int(id)-1][1]*224
        box[3]=box[3]/id_to_size[int(id)-1][0]*224
        id_to_box[int(id)] = box
id_to_box = np.array(list(id_to_box.values()))

index=[i for i in range(11788)]
index=random.sample(index,100)


model=keras.models.load_model("model.h5")
result=model.predict(id_to_data[index,:,:,:])

mean=[0.485,0.456,0.406]
std=[0.229,0.224,0.225]
j=0
for i in index:
    print("Predicting "+str(i)+"th image.")
    true_box=boid_to_datax[i]
    image=id_to_data[i]
    prediction=result[j]
    j+=1
    for channel in range(3):
        image[:,:,channel]=image[:,:,channel]*std[channel]+mean[channel]

    image=image*255
    image=image.astype(np.uint8)
    plt.imshow(image)
    plt.gca().add_patch(plt.Rectangle((true_box[0],true_box[1]),true_box[2],true_box[3],fill=False,edgecolor='red',linewidth=2,alpha=0.5))
    plt.gca().add_patch(plt.Rectangle((prediction[0]*224,prediction[1]*224),prediction[2]*224,prediction[3]*224,fill=False,edgecolor='green',linewidth=2,alpha=0.5))
    plt.show()
    #plt.savefig("./prediction/"+str(i)+".png")
    plt.cla()



