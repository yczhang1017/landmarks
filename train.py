
import os
import shutil
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

import torch.utils.data
import torch.utils.data.distributed

import time
import argparse
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
#from torchvision.datasets import ImageFolder
#from CNNs import CNN_models
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


parser = argparse.ArgumentParser(
    description='Google Landmarks Recognition')
parser.add_argument('--data', metavar='DIR',default='./compress',
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
'''---------DALI------------'''
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--dali_cpu', action='store_false',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument('-t', '--test', action='store_true',
                    help='Launch test mode with preset arguments')
parser.add_argument("--local_rank", default=0, type=int)


cudnn.benchmark = True
NLABEL=203093
PRIMES=[491,499]
mean=[108.8230125, 122.87493125, 130.4728]
std=[62.5754482, 65.80653705, 79.94356993]
args = parser.parse_args()
best_prec1 = 0

args.distributed = False
if args.fp16 or args.distributed:
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")

def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]        

if args.test:
    args.fp16 = False
    args.epochs = 1
    args.start_epoch = 0
    args.arch = 'resnet50'
    args.batch_size = 64
    args.data = []
    args.prof = True
    args.data.append('/test')
    args.data.append('/test')

args.distributed = False
args.gpu = 0
args.world_size = 1
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1


    
def id2path(root,id):
    return os.path.join(root,id[0],id[1],id[2],id+'.jpg')

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu=False, file_list):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=True, file_list=file_list)
        if dali_cpu:
            dali_device = "cpu"
            self.decode = ops.HostDecoderRandomCrop(device=dali_device, output_type=types.RGB,
                                                    random_aspect_ratio=[0.8, 1.25],
                                                    random_area=[0.1, 1.0],
                                                    num_attempts=100)
        else:
            dali_device = "gpu"
            self.decode = ops.nvJPEGDecoderRandomCrop(device="mixed", output_type=types.RGB, device_memory_padding=211025920, host_memory_padding=140544512,
                                                      random_aspect_ratio=[0.8, 1.25],
                                                      random_area=[0.1, 1.0],
                                                      num_attempts=100)
        self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std)
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]
    
class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, file_list):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False, file_list=file_list)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
    
    
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
        '''if img.width < img.height:
            s1=random.randint(0,img.height-img.width)
            img=img.crop((0,s1,img.width,s1+img.width))
        else:
            s1=random.randint(0,img.width-img.height)
            img=img.crop((s1,0,s1+img.height,img.height))'''

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
    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
        
    if not os.path.exists(args.save_folder):
        os.mkdir(args.save_folder)
         
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
    print('Number of labels:',df_count.shape[0])
    print('We sampled ',rs,'starting from label',label_start,'as validation data')
    
    
    labels=dict()
    labels['val']=df2['label'].sample(n=rs)
    labels['train']=df['label'].drop(labels['val'].index)
    
    
    txt_path=dict()
    for phase in ['train','val']:
        txt_path[phase]=os.path.join(args.data,phase+'.txt')
        file1 = open(txt_path[phase],"w") 
        lc1=labels[phase].index.tolist()
        lc2=labels[phase].tolist()
        for id,ll in zip(lc1,lc2):
            file1.write(id[0]+'/'+id[1]+'/'+id[2]+'/'+id+'.jpg'+' '+str(ll)+'\n')
        file1.close()
    
    crop_size = 224
    val_size = 256
    dataloader=dict()
    if args.dali_cpu:
        print('use CPU to load data')
    else:
        print('use GPU to load data')
        
    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank,
                           data_dir=args.data, crop=crop_size, dali_cpu=args.dali_cpu, file_list=txt_path['train'])
    pipe.build()
    dataloader['train'] = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    
    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
                         data_dir=args.data, crop=crop_size, size=val_size, file_list=txt_path['train'])
    pipe.build()
    dataloader['val'] = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    
    
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
            
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer=[]
    scheduler=[]
    for i,p in enumerate(PRIMES):
        optimizer.append(optim.SGD(model[i].parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay))
        if args.fp16:
            optimizer[i] = FP16_Optimizer(optimizer[i],
                     static_loss_scale=args.static_loss_scale,
                     dynamic_loss_scale=args.dynamic_loss_scale)
        scheduler.append(optim.lr_scheduler.StepLR(optimizer[i], step_size=args.step_size, gamma=0.1))
        for i in range(args.resume_epoch):
            scheduler[i].step()
    
    best_acc=0    
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
            end = time.time()
            for nb, (inputs,targets) in enumerate(dataloader[phase]):
                data_time=time.time()-end
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
                batch_time=time.time()-end
                end=time.time()
                if (nb+1) % args.print_freq ==0:
                    print('{} L:{:.4f} correct:{:.0f} acc1: {:.4f} Time: {:.4f}s {:.4f}s'
                          .format(num,cur_avg_loss,csum,acc1,data_time,batch_time))
                    cur=0
                    cur_loss=0.0
        
            print('------SUMMARY:',phase,'---------')
            print('{} L:{:.4f} correct:{:.0f} acc1: {:.4f} Time: {:.4f}s'
                      .format(num,average_loss,csum,acc1,batch_time))
            dataloader[phase].reset()
            if phase == 'val':
                save_dict=dict()
                for i,p in enumerate(PRIMES):
                    save_dict[p]= model[i].state_dict()    
                
                save_dict['epoch']= epoch + 1
                save_dict['acc'] = acc1
                save_dict['arch']=args.arch
                save_file=os.path.join(args.save_folder,'checkpoint_'+str(epoch+1)+'.pth')
                torch.save(save_dict,save_file)
                if acc1>best_acc:
                    shutil.copyfile(save_file, 'model_best.pth.tar')
                
if __name__ == '__main__':
    main()   
    
    