import os
import errno
import numpy as np
import PIL
import pandas as pd

csv_file=os.path.join('./','train.csv')
df=pd.read_csv(csv_file,index_col=0)
print('csv loaded')
image_dir=os.path.join('./','train')
save_dir=os.path.join('./','compress')

df=df.drop(['url'], axis=1)
num=0

for id,_ in df.iterrows():
    path=os.path.join(save_dir,id[0],id[1],id[2])
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
    
    in_path=os.path.join(image_dir,id[0],id[1],id[2],id+'.jpg')
    out_path=os.path.join(save_dir,id[0],id[1],id[2],id+'.jpg')
    
    img=PIL.Image.open(in_path)
    if img.width < img.height:
        s1=(img.height-img.width)/2
        img=img.crop((0,s1,img.width,s1+img.width))
    else:
        s1=(img.width-img.height)/2
        img=img.crop((s1,0,s1+img.height,img.height))
    
    img=img.resize((256,256), resample=PIL.Image.LANCZOS)
    img.save(out_path)
    num+=1
    if num%100==0:
        print(num)