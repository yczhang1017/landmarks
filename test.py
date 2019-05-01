
import os
import pickle
import torch

import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.utils.data.distributed
import time
import argparse

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

parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch_size', default=256, type=int,
                    metavar='N',
                    help='Batch size for training')
parser.add_argument('--checkpoint', default=None,  type=str, metavar='PATH',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('-s','--save_folder', default='save/', type=str,
                    help='Dir to save results')


parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')

from train import PRIMES,HybridValPipe
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
    
'''
class HybridTestPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size, file_list):
        super(HybridTestPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=args.local_rank, num_shards=args.world_size, random_shuffle=False, file_list=file_list)
        self.decode = ops.nvJPEGDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std)
    def define_graph(self):
        jpegs = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return output
'''    
def main():
    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
    
    txt_path=os.path.join(args.data,'file_list.txt')
    file1 = open(txt_path,"w")
    image_ids=[]
    for jpg in os.listdir(args.data):
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
    
    
    
    model=[None]*len(PRIMES)
    for i,p in enumerate(PRIMES):
        model[i]=models.__dict__[args.arch](num_classes=p)
        check_file=os.path.join(args.data,args.checkpoint)
        if args.checkpoint:
            model[i].load_state_dict(torch.load(check_file['state_'+str(p)],
                                 map_location=lambda storage, loc: storage))
        if torch.cuda.is_available():
            model[i] = model[i].cuda(device)
            if args.fp16:
                model[i] = network_to_half(model[i])
        model[i].eval()
    print('Finished loading model!')
    output_file=os.path.join('./','results.csv')
    f1=open("label_id.pkl","rb")
    label2id=pickle.load(f1)
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
        for ib, data in enumerate(dataloader):
            inputs = data[0]["data"].to(device, non_blocking=True)
            sublabel=torch.zeros((inputs.size(0),len(PRIMES)),dtype=torch.int64)
            preds=torch.zeros((inputs.size(0)),dtype=torch.int64).cpu()
            
            for i,p in enumerate(PRIMES):
                outputs=model[i](inputs)
                sublabel[:,i] = outputs.argmax(dim=1)
                if i>0:
                    preds[:,i-1]=(sublabel[:,0]-sublabel[:,i]).cpu().numpy()
                    preds.apply_(tolabel)
                    preds=preds+sublabel[:,i]
                    
                count=inputs.shape[0]
                for j in range(count):
                    f.write(image_ids[ii]+','+str(label2id[preds[j].item()])+'\n')
                    ii=ii+1
            t01= t02
            t02= time.time()
            dt1=(t02-t01)/count
            if (ib+1)%10==0:
                print('Image {:d}/{:d} time: {:.4f}s'.format(ii+1,total,dt1))
            
    f.close()
                
if __name__ == '__main__':
    main()   
    
    