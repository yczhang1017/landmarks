
import os
import pickle
import torch
import PIL

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import time
import argparse
from torchvision.transforms import transforms
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))




parser = argparse.ArgumentParser(
    description='Google Landmarks Recognition')
parser.add_argument('--data', metavar='DIR',default='./test2',
                    help='path to dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=512, type=int,
                    metavar='N',
                    help='Batch size for training')
parser.add_argument('-c','--checkpoint', default=None,  type=str, metavar='PATH',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('-s','--save_folder', default='save/', type=str,
                    help='Dir to save results')


parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')

NLABEL=203094
PRIMES=[491,499]
mean=[108.8230125, 122.87493125, 130.4728]
std=[62.5754482, 65.80653705, 79.94356993]
cudnn.benchmark = True
args = parser.parse_args()
best_prec1 = 0

args.fp16=False
args.distributed = False


args.distributed = False
args.gpu = 0
args.local_rank=0
args.world_size=1
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
    

class TestDataset(torch.utils.data.Dataset):
    def __init__(self,root,image_labels=None ,transform=None):
        self.root=os.path.expanduser(root)
        self.transform = transform
        self.image_labels=image_labels
        
        
    def __getitem__(self, idx):
        img_id=self.image_labels[idx]
        img_path=os.path.join(self.root,img_id)
        img=PIL.Image.open(img_path)
        if self.transform is not None:
            im_tensor = self.transform(img)
        return im_tensor
    
    def __len__(self):
        return len(self.image_labels)

def main():
    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
    
    
    image_ids=os.listdir(args.data)
    transform=transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean,std),
        ])
    
    dataset=TestDataset(args.data,image_labels=image_ids,transform=transform) 
            
    dataloader=torch.utils.data.DataLoader(dataset,
            batch_size=args.batch_size,shuffle=False,num_workers=args.workers,pin_memory=True)
            
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    
    
    model=[None]*len(PRIMES)
    for i,p in enumerate(PRIMES):
        model[i]=models.__dict__[args.arch](num_classes=p)
        if args.checkpoint:
            model[i].load_state_dict(
                    torch.load(args.checkpoint,map_location=lambda storage, loc: storage.cuda(args.gpu))
                    ['state_'+str(p)])
        if torch.cuda.is_available():
            model[i] = model[i].cuda(device)
        model[i].eval()
    print('Finished loading model!')
    output_file=os.path.join('./','results.csv')
    f1=open("label_id.pkl","rb")
    label2id=pickle.load(f1)
    maxlabel=sorted(label2id.keys())[-1]
    
    f1.close()
    f=open( output_file , mode='w+')
    f.write('id,landmarks\n')
    
    
    p0=PRIMES[0]
    p1=PRIMES[1]

    dp=p1-p0
    res=[]
    for j in range(dp):
        res.append((-p0*j)%dp)
        
    def tolabel(i):
        j=res.index(i%dp)
        return (i+j*p0)//dp*p1
        
    t02=time.time()
    ii=0
    total=len(image_ids)
    with torch.no_grad():
        for ib, inputs in enumerate(dataloader):
            inputs = inputs.to(device, non_blocking=True)
            sublabel=torch.zeros((inputs.size(0),len(PRIMES)),dtype=torch.int64)
            preds=torch.zeros((inputs.size(0)),dtype=torch.int64).cpu()
            count=inputs.shape[0]
            for i,p in enumerate(PRIMES):
                outputs=model[i](inputs)
                sublabel[:,i] = outputs.argmax(dim=1)
                
            preds=((sublabel[:,0]-sublabel[:,1])%p0).cpu()
            preds.apply_(tolabel)
            preds=preds+sublabel[:,1]
            
            for j in range(count):
                label=preds[j].item()
                if label > maxlabel:
                    _,pros =outputs[j,:].topk(20)
                    pros=pros.cpu()
                    k=1
                while label > maxlabel:
                    preds_j=(sublabel[j,0]-pros[k])%p0
                    label=tolabel(preds_j.item())+pros[k].item()
                    k=k+1
                f.write(image_ids[ii].split('.')[0]+','+str(label2id[label])+'\n')
                ii=ii+1
            t01= t02
            t02= time.time()
            dt1=(t02-t01)
            if (ib+1)%10==0:
                print('Image {:d}/{:d} time: {:.4f}s'.format(ii,total,dt1))
            
    f.close()
                
if __name__ == '__main__':
    main()   
    
    