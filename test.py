
import os
import pickle
import torch
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import time
import argparse
#import rnet
import snet as mynet


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
parser.add_argument('--data', metavar='DIR',default='./test',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='s_resnext50_32x4d',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: s_resnext50_32x4d)')
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


from train import NLABEL,PRIMES,mean,std
from train import HybridValPipe
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
    
    df=pd.read_csv('test.csv',index_col=0)
    ids=df.index.tolist()
    
    for id in ids:
        file1.write(id[0]+'/'+id[1]+'/'+id[2]+'/'+id+'.jpg 0\n')
        image_ids.append(id)
    file1.close()
    
    crop_size =224
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
    else:
        model=mynet.__dict__[args.arch](pretrained=None,num_classes=PRIMES)
        model.load_state_dict(torch.load(args.checkpoint,
                map_location=lambda storage, loc: storage)['state'])
        if torch.cuda.is_available():
                model = model.cuda(device)
                
    print('Finished loading model!')
    f1=open("label_id.pkl","rb")
    label2id=pickle.load(f1)
    maxlabel=sorted(label2id.keys())[-1]
    assert maxlabel+1 == NLABEL
    f1.close()
    
    '''
    sample_path='recognition_sample_submission.csv'
    df=pd.read_csv(sample_path,index_col=0)
    nones=df.index.drop(image_ids)
    '''
    
    p0=PRIMES[0]
    p1=PRIMES[1]

    dp=p1-p0
    res=[]
    of=open("results.csv",mode='w+')
    of.write('id,landmarks\n')
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
            nn=inputs.shape[0]
            
            softmax=torch.nn.Softmax(dim=1)
            if args.arch in model_names:
                outputs=model[0](inputs)
                scores=softmax(outputs).unsqueeze(2).expand((nn,p0,p1))
                
                outputs=model[1](inputs)
                scores= scores * softmax(outputs).unsqueeze(1).expand((nn,p0,p1))
                
            else:
                outputs=model(inputs)
                scores=softmax(outputs[0]).unsqueeze(2).expand((nn,p0,p1))
                scores= scores * softmax(outputs[1]).unsqueeze(1).expand((nn,p0,p1))
                scores= scores.reshape((scores.shape[0],p0*p1))
                scores[:,NLABEL:]=0
                scores = scores/scores.sum(dim=1,keepdim=True)
                conf, pred = scores.max(dim=1)
                conf=conf.cpu()
                pred=pred.cpu()
                for j in range(nn):
                    if ii< len(ids):
                        of.write('{:s},{:d} {:.6f}\n'.format(ids[ii].split('.')[0],
                                 label2id[pred[j].item()],conf[j].item()))
                        ii=ii+1
                        
            t01= t02
            t02= time.time()
            dt1=(t02-t01)
            if (ib+1)%10==0:
                print('Image {:d}/{:d} time: {:.4f}s'.format(ii,total,dt1))
    of.close()
                
if __name__ == '__main__':
    main()   
    
    