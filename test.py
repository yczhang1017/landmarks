
import os
import pickle
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import time
import argparse
import rnet
from rnet import NLABEL,PRIMES,mean,std

import torchvision.models as models
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
parser.add_argument('--data', metavar='DIR',default='./test2',
                    help='path to dataset')

parser.add_argument('-a', '--arch', metavar='ARCH', default='rnet34',
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

from train import HybridValPipe
cudnn.benchmark = True
args = parser.parse_args()
best_prec1 = 0

args.fp16=False
args.distributed = False
if args.fp16 or args.distributed:
    try:
        #from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import FP16_Optimizer,network_to_half
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")


args.distributed = False
args.gpu = 0
args.local_rank=0
args.world_size=1
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1
    

def main():
    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
    
    txt_path=os.path.join(args.data,'file_list.txt')
    file1 = open(txt_path,"w")
    image_ids=[]

    for jpg in os.listdir(args.data):
        if jpg.endswith('.jpg'):
            file1.write(jpg+' 0\n')
            image_ids.append(jpg.split('.')[0])
    file1.close()
    
    crop_size = 224
    val_size = 256
    
    
    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
                         data_dir=args.data, crop=crop_size, size=val_size, file_list=txt_path)
    pipe.build()
    dataloader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    
     
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    
    if args.arch in model_names:
        model=[None]*len(PRIMES)
        for i,p in enumerate(PRIMES):
            model[i]=models.__dict__[args.arch](num_classes=p)
            if args.checkpoint:
                model[i].load_state_dict(torch.load(args.checkpoint,
                    map_location=lambda storage, loc: storage.cuda(args.gpu))['state_'+str(p)])
            if torch.cuda.is_available():
                model[i] = model[i].cuda(device)
                if args.fp16:
                    model[i] = network_to_half(model[i])
            model[i].eval()
    elif args.arch in rnet.__dict__:
        model=rnet.__dict__[args.arch](pretrained=False,num_classes=PRIMES)
        model.load_state_dict(torch.load(args.checkpoint,
                map_location=lambda storage, loc: storage)['state'])
    print('Finished loading model!')
    f1=open("label_id.pkl","rb")
    label2id=pickle.load(f1)
    maxlabel=sorted(label2id.keys())[-1]
    
    f1.close()
    
    
    sample_path='recognition_sample_submission.csv'
    df=pd.read_csv(sample_path,index_col=0)
    nones=df.index.drop(image_ids)
    
    p0=PRIMES[0]
    p1=PRIMES[1]
    results=[]
    confidence=[]
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
        for ib, data in enumerate(dataloader):
            inputs = data[0]["data"].to(device, non_blocking=True)
            sublabel=torch.zeros((inputs.size(0),len(PRIMES)),dtype=torch.int64)
            subscore=torch.zeros((inputs.size(0),len(PRIMES)),dtype=torch.float)
            
            preds=torch.zeros((inputs.size(0)),dtype=torch.int64).cpu()
            score=torch.zeros((inputs.size(0)),dtype=torch.float).cpu()
            count=inputs.shape[0]
            if args.arch in model_names:
                for i,p in enumerate(PRIMES):
                    outputs=model[i](inputs)
                    subscore[:,i],sublabel[:,i] = outputs.max(dim=1)
                    
            elif args.arch in rnet.__dict__:
                outputs=model(inputs)
                for i,p in enumerate(PRIMES):
                    subscore[:,i],sublabel[:,i] = outputs[i].max(dim=1)
                
                
            for i,p in enumerate(PRIMES):
                if i>0:
                    preds=((sublabel[:,0]-sublabel[:,i])%p0).cpu()
                    preds.apply_(tolabel)
                    preds=preds+sublabel[:,i]
                    
                    for j in range(count):
                        label=preds[j].item()
                        if label > maxlabel:
                            scoj,pros =outputs[j,:].topk(20)
                            pros=pros.cpu()
                            k=0
                            while label > maxlabel:
                                k=k+1
                                preds_j=(sublabel[j,0]-pros[k])%p0
                                label=tolabel(preds_j.item())+pros[k].item()
                            subscore[j,1]=scoj[k]
                        results.append(label2id[label])
                        ii=ii+1
                        
                    score=subscore[:,0]*subscore[:,1]
                    confidence=confidence+score.tolist()
                        
                        
            t01= t02
            t02= time.time()
            dt1=(t02-t01)
            if (ib+1)%10==0:
                print('Image {:d}/{:d} time: {:.4f}s'.format(ii,total,dt1))
    
    
    results=results[:len(image_ids)]
    confidence=confidence[:len(image_ids)]
    print('number of detected labels: ',len(results))
    df.loc[image_ids,'landmarks']=[str(r)+' '+str(c) for r,c in zip(results,confidence)]
    detected = pd.DataFrame({'landmarks':results}, index =image_ids) 
    most=detected.groupby('landmarks').size().idxmax()
    
    fornone=str(most)+' 0.001'
    print('nones are asigned: ',fornone)
    df.loc[nones,'landmarks']=fornone
    df.to_csv(path_or_buf='results.csv')
                
if __name__ == '__main__':
    main()   
    
    