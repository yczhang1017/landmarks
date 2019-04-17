import os
import numpy as np
import pandas as pd
from scipy.misc import imread


csv_file=os.path.join('../','test.csv')
df=pd.read_csv(csv_file)
NN=df.shape[0]
image_dir=os.path.join('../','train')
#colors=['blue','red','yellow','green'];mode='CMYK'
all_mean=np.zeros(3)
all_std=np.zeros(3)
#images=list(label_dict.keys())
#randl=np.random.choice(len(images), int(len(images)/5), replace=False)#
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
'''
for color in colors:
    all_mean=0
    all_var=0
    for i,ind in enumerate(randl):
        img=images[ind]
        img_l4=[]
        image=os.path.join(image_dir,img+'_'+color+'.png')
            
        img4=PIL.Image.open(image)
        
        transform= transforms.Compose(
                    [transforms.ToTensor()])       
        img4t=transform(img4)     
        all_mean+=img4t.mean().item()
        all_var+=img4t.var().item()
        if (i+1)%300==0:
            print(i,len(label_dict),all_mean/(i+1),all_var/(i+1))
    mean.append(all_mean/len(randl))
    std.append(np.sqrt(all_var/len(randl)))
'''
'''   
with open('mean_std.pkl', 'w') as f:
    pickle.dump(list(mean.values()), f)
    pickle.dump(list(std.values()), f)
'''

'''dim=img4t.shape[0]
    if i==0:
        image_means =mean.view(1,dim)
        image_vars =mean.view(1,dim)
    else:
        image_means=torch.cat([image_means,mean.view(1,dim)],0)
        image_vars=torch.cat([image_vars,mean.view(1,dim)],0)
all_mean=image_means.mean(0)       
all_std=image_vars.mean(0).sqrt()  
print(all_mean,all_std)'''
