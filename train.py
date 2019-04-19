
import os
import numpy as np
import pandas as pd


#from imageio import imread
import random
import PIL
#from augmentations import PhotometricDistort

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.transforms import transforms
#from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
import argparse
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
#from torchvision.datasets import ImageFolder
#from CNNs import CNN_models


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(
    description='Google Landmarks Recognition')
parser.add_argument('--data', metavar='DIR',default='./train',
                    help='path to dataset')
parser.add_argument('-s','--save_folder', default='save/', type=str,
                    help='Dir to save results')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e','--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=32, type=int,
                    metavar='N',
                    help='Batch size for training')
parser.add_argument('-lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-w','--weight_decay', default=5e-4, type=float,
                    help='Weight decay')



parser.add_argument('--step_size', default=5, type=int,
                    help='Number of steps for every learning rate decay')
parser.add_argument('--checkpoint', default=None,  type=str, metavar='PATH',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='epoch number to be resumed at')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='Use pre-trained weights')

NLABEL=203093
PRIMES=[491,499]

def id2path(root,id):
    return os.path.join(root,id[0],id[1],id[2],id+'.jpg')

class LandmarksDataset(torch.utils.data.Dataset):
    def __init__(self,root,phase,image_labels=None, size=224 ,transform=None):
        self.root=os.path.expanduser(root)
        self.phase=phase
        self.transform = transform
        self.size=size
        self.image_labels=image_labels
        
        
    def __getitem__(self, idx):
        if self.phase in ['train','val']:
            img_id=self.image_labels.index[idx]
            label=self.image_labels.iloc[idx]
            #img_path,label=self.image_labels[index]
        elif self.phase in ['test']:
            img_id=self.image_labels.index[idx]
            #img_path=self.image_labels[idx]
        
        img_path=id2path(self.root,img_id)
        img=PIL.Image.open(img_path)
        '''random crop the image based the shorter side'''
        if img.width < img.height:
            s1=random.randint(0,img.height-img.width)
            img=img.crop((0,s1,img.width,s1+img.width))
        else:
            s1=random.randint(0,img.width-img.height)
            img=img.crop((s1,0,s1+img.height,img.height))

        if self.transform is not None:
            im_tensor = self.transform(img)
        
        if self.phase in ['train','val']:
            target=[]
            for p in PRIMES:
                target.append(label%p)
            #target=torch.tensor(target,device="cpu")
            return im_tensor,target
        elif self.phase in ['test']:
            return im_tensor
    
    def __len__(self):
        return len(self.image_labels)

    
def main():
    args = parser.parse_args()
    
    mean=[108.8230125, 122.87493125, 130.4728]
    std=[62.5754482, 65.80653705, 79.94356993]
    transform={
    'train': transforms.Compose([
         transforms.Resize(256),
         #transforms.RandomAffine(20,shear=20,resample=PIL.Image.BILINEAR),
         #transforms.RandomRotation(20),
         transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean,std),
         ]),
    'val':transforms.Compose([
         transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize(mean,std)
         ])}
    
    
    csv_file=os.path.join(args.data,'train.csv')
    df=pd.read_csv(csv_file,index_col=0)
    df=df.drop(['url'], axis=1)
    
    df_count=df.groupby('landmark_id').size()
    df_count=df_count.sort_values()
    df_count=df_count.to_frame('count')
    df_count['label'] = np.arange(len(df_count))
    label_dict=df_count.loc[:,'label'].to_dict()    
    
    df['label']=df['landmark_id'].map(label_dict)
    label_start=df_count[df_count['count']>2].iloc[0,1]
    df2=df.loc[df['label']>label_start]
    
    r=df2.shape[0]
    rs=np.int(r/30)
    print('Number of labels:',df_count.size)
    print('We sampled ',rs,'starting from label',label_start,'as validation data')
    
    
    labels=dict()
    labels['val']=df2['label'].sample(n=rs)
    labels['train']=df['label']#.drop(labels['val'].index)
    
    dataset={x: LandmarksDataset(args.data,x,labels[x],transform=transform[x]) 
            for x in ['train', 'val']}
    dataloader={x: torch.utils.data.DataLoader(dataset[x],
            batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
            for x in ['train', 'val']}
    
    model=[]
    
    if torch.cuda.is_available():
        #torch.set_default_tensor_type(torch.float32)
        device = torch.device("cuda:0")
    else:
        #torch.set_default_tensor_type(torch.float32)
        device = torch.device("cpu")
    
    for i,p in enumerate(PRIMES):
        model.append(models.__dict__[args.arch](num_classes=p))
        if (not args.checkpoint) and args.pretrained:
            model_type=''.join([i for i in args.arch if not i.isdigit()])
            model_url=models.__dict__[model_type].model_urls[args.arch]
            pre_trained=model_zoo.load_url(model_url)
            pre_trained['fc.weight']=pre_trained['fc.weight'][:p,:]
            pre_trained['fc.bias']=pre_trained['fc.bias'][:p]   
            model[i].load_state_dict(pre_trained)
        if torch.cuda.is_available():
            model[i] = model[i].cuda()
            
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer=[]
    scheduler=[]
    for i,p in enumerate(PRIMES):
        optimizer.append(optim.SGD(model[i].parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay))
        scheduler.append(optim.lr_scheduler.StepLR(optimizer[i], step_size=args.step_size, gamma=0.1))
        for i in range(args.resume_epoch):
            scheduler[i].step()
        
    for epoch in range(args.resume_epoch,args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 5)
        for phase in ['train','val']:    
            if phase == 'train':
                for i,p in enumerate(PRIMES):    
                    scheduler[i].step()
                    model[i].train()
            else:
                for i,p in enumerate(PRIMES):    
                    model[i].eval()
            
            num=0 
            csum=0       
            
            for inputs,targets in dataloader[phase]:
                t01 = time.time()
                inputs = inputs.to(device) 
                for i,p in enumerate(PRIMES):
                    targets[i]= targets[i].to(device)
                    optimizer[i].zero_grad()
                
                
                batch_size = inputs.size(0)
                correct=torch.ones((batch_size,1),dtype=torch.uint8)
                with torch.set_grad_enabled(phase == 'train'):
                    for i,p in enumerate(PRIMES):
                        outputs=model[i](inputs)
                        loss = criterion(outputs, targets[i])
                        if phase == 'train':
                            loss.backward()
                            optimizer[i].step()
                        '''calcaulate accuracy'''
                        _, pred = outputs.topk(1, 1, True, True)
                        pred=pre.t()
                        correct = correct*pred.eq(targets[i].view(1, -1))
                
                print('correct shape',correct.shape)
                num+=batch_size
                csum+=correct.view(-1).float().sum(0, keepdim=True)
                acc1= csum/num*100
                t02 = time.time()
                print(phase,':')
                print('{} correct:{:.0f} acc1: {:.4f} Time: {:.4f}s'.format(num,csum,acc1,t02-t01))
                                
            
 
if __name__ == '__main__':
    main()   
    
    