
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
parser.add_argument('-s','--save_folder', default='checkpoints/', type=str,
                    help='Dir to save results')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-e','--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='Batch size for training')
parser.add_argument('-lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('-w','--weight_decay', default=1e-4, type=float,
                    help='Weight decay')

parser.add_argument('--step_size', default=5, type=int,
                    help='Number of steps for every learning rate decay')
parser.add_argument('--checkpoint', default=None,  type=str, metavar='PATH',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='epoch number to be resumed at')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='Use pre-trained weights')


parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')

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
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
        
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
    df2=df.loc[df['label']>=label_start]
    
    r=df2.shape[0]
    rs=np.int(r/50)
    print('Number of images:',df.shape[0])
    print('Number of labels:',df_count.size)
    print('We sampled ',rs,'starting from label',label_start,'as validation data')
    
    
    labels=dict()
    labels['val']=df2['label'].sample(n=rs)
    labels['train']=df['label'].drop(labels['val'].index)
    
    dataset={x: LandmarksDataset(args.data,x,labels[x],transform=transform[x]) 
            for x in ['train', 'val']}
    dataloader={x: torch.utils.data.DataLoader(dataset[x],
            batch_size=args.batch_size,shuffle=True,num_workers=args.workers,pin_memory=True)
            for x in ['train', 'val']}
    
    model=[]
    
    if torch.cuda.is_available():
        #torch.set_default_tensor_type(torch.float32)
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
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
            model[i] = model[i].cuda(device)
            
    criterion = nn.CrossEntropyLoss().cuda(device)
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
            running_loss=0.0
            cur=0
            cur_loss=0.0
            print(phase,':')
            t02=0
            for nb, (inputs,targets) in enumerate(dataloader[phase]):
                t20 = t02 
                t01 = time.time()
                inputs = inputs.to(device, non_blocking=True)
                for i,p in enumerate(PRIMES):
                    targets[i]= targets[i].to(device, non_blocking=True)
                    optimizer[i].zero_grad()
                
                batch_size = inputs.size(0)
                correct=torch.ones((batch_size),dtype=torch.uint8).to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    for i,p in enumerate(PRIMES):
                        outputs=model[i](inputs)
                        loss = criterion(outputs, targets[i])
                        if phase == 'train':
                            loss.backward()
                            optimizer[i].step()
                        _, pred = outputs.topk(1, 1, True, True)
                        correct = correct.mul(pred.view(-1).eq(targets[i]))
                        
                
                num+=batch_size
                csum+=correct.float().sum(0)
                acc1= csum/num*100
                running_loss += loss.item() * batch_size
                average_loss=running_loss/num
                cur+=batch_size
                cur_loss+=loss.item() * batch_size
                cur_avg_loss=cur_loss/cur
                t02 = time.time()    
                if (nb+1) % args.print_freq ==0:
                    print('{} L:{:.4f} correct:{:.0f} acc1: {:.4f} Time: {:.4f}s {:.4f}s'
                          .format(num,cur_avg_loss,csum,acc1,t02-t01,t02-t20))
                    cur=0
                    cur_loss=0.0
        
            print('------SUMMARY:',phase,'---------')
            print('{} L:{:.4f} correct:{:.0f} acc1: {:.4f} Time: {:.4f}s'
                      .format(num,average_loss,csum,acc1,t02-t01))
            if phase == 'val':
                for i,p in enumerate(PRIMES):
                    torch.save(model[i].state_dict(),os.path.join(args.save_folder,'w'+str(i)+'_'+str(epoch+1)+'.pth'))
 
if __name__ == '__main__':
    main()   
    
    