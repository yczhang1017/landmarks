import os
import numpy as np
import pandas as pd
from imageio import imread


csv_file=os.path.join('../','train.csv')
df=pd.read_csv(csv_file)
NN=df.shape[0]
image_dir=os.path.join('../','train')

all_mean=np.zeros(3)
all_std=np.zeros(3)

mm=int(NN/10)
randl=np.random.choice(NN,mm,replace=False)


for i,ind in enumerate(randl):
    id=df.at[ind,'id']
    image_path=os.path.join(image_dir,id[0],id[1],id[2],id+'.jpg')
    img3=imread(image_path)
    std=np.std(np.std(img3, axis=0),axis=0)
    mean=np.mean(np.std(img3, axis=0),axis=0)
    all_mean+=mean
    all_std+=std

all_mean/=mm
all_std/=mm
print(mean)
print(std)