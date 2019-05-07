
import os
import shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torch.utils.data
import torch.utils.data.distributed

import time
import argparse
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
import rnet
from rnet import NLABEL,PRIMES,mean,std


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

parser.add_argument('-a', '--arch', metavar='ARCH', default='rnet34',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: rnet34)')
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

parser.add_argument('--step_size', default=10, type=int,
                    help='Number of steps for every learning rate decay')
parser.add_argument('--checkpoint', default=None,  type=str, metavar='PATH',
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--resume_epoch', default=0, type=int,
                    help='epoch number to be resumed at')
parser.add_argument('-s','--save_folder', default='save/', type=str,
                    help='Dir to save results')


parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
'''---------DALI------------'''
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')
parser.add_argument('--dali_cpu', action='store_true',
                    help='Runs CPU based version of DALI pipeline.')
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument('--prof', dest='prof', action='store_true',
                    help='Only run 10 iterations for profiling.')
parser.add_argument("--local_rank", default=0, type=int)


cudnn.benchmark = True
args = parser.parse_args()
best_prec1 = 0

args.distributed = False
if args.fp16 or args.distributed:
    try:
        #from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import FP16_Optimizer,network_to_half
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this example.")
    


args.distributed = False
args.gpu = 0
args.world_size = 1
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, dali_cpu, file_list):
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
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, labels]
    
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
                                            mean=mean,
                                            std=std)
    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, labels]
    

        
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
    del df,df_count,df2,labels,label_dict
    
    crop_size = 224
    val_size = 256
    dataloader=dict()
    
    print('use '+['GPU','CPU'][args.dali_cpu]+' to load data')
    print('Half precision:'+str(args.fp16))
    
    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank,
                           data_dir=args.data, crop=crop_size, dali_cpu=args.dali_cpu, file_list=txt_path['train'])
    pipe.build()
    dataloader['train'] = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    
    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
                         data_dir=args.data, crop=crop_size, size=val_size, file_list=txt_path['val'])
    pipe.build()
    dataloader['val'] = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))
    
    
    model=[]
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    
    
    criterion = nn.CrossEntropyLoss().cuda()
    model=[None]*len(PRIMES)
    optimizer=[None]*len(PRIMES)
    scheduler=[None]*len(PRIMES)
    
    if args.arch in model_names:
        for i,p in enumerate(PRIMES):
            model[i]=models.__dict__[args.arch](num_classes=p)
            if not args.checkpoint:
                model_type=''.join([i for i in args.arch if not i.isdigit()])
                model_url=models.__dict__[model_type].model_urls[args.arch]
                pre_trained=model_zoo.load_url(model_url)
                pre_trained['fc.weight']=pre_trained['fc.weight'][:p,:]
                pre_trained['fc.bias']=pre_trained['fc.bias'][:p]   
                model[i].load_state_dict(pre_trained)
            elif args.checkpoint:
                print('Resuming training from epoch {}, loading {}...'
                  .format(args.resume_epoch,args.checkpoint))
                check_file=os.path.join(args.data,args.checkpoint)
                model[i].load_state_dict(torch.load(check_file['state_'+str(p)],
                                     map_location=lambda storage, loc: storage))
            if torch.cuda.is_available():
                model[i] = model[i].cuda(device)
                if args.fp16:
                    model[i] = network_to_half(model[i])
        for i,p in enumerate(PRIMES):
            optimizer[i]=optim.SGD(model[i].parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
            if args.checkpoint:
                check_file=os.path.join(args.data,args.checkpoint)
                optimizer[i].load_state_dict(torch.load(check_file['optim_'+str(p)],
                                     map_location=lambda storage, loc: storage))
            scheduler[i]=optim.lr_scheduler.StepLR(optimizer[i], step_size=args.step_size, gamma=0.1)
            for i in range(args.resume_epoch):
                scheduler[i].step()       
    elif args.arch in rnet.__dict__:
        if args.checkpoint:
            model=rnet.__dict__[args.arch](pretrained=False,num_classes=PRIMES)
            check_file=os.path.join(args.data,args.checkpoint)
            model.load_state_dict(torch.load(check_file['state'],
                             map_location=lambda storage, loc: storage))
        else:
            model=rnet.__dict__[args.arch](pretrained=True,num_classes=PRIMES)
        
        if torch.cuda.is_available():
            model = model.cuda(device)
        optimizer=optim.SGD(model.parameters(),lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
        if args.checkpoint:
                check_file=os.path.join(args.data,args.checkpoint)
                optimizer.load_state_dict(torch.load(check_file['optim'],
                                     map_location=lambda storage, loc: storage))
        scheduler=optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
        for i in range(args.resume_epoch):
            scheduler.step()       
    
    
    best_acc=0    
    for epoch in range(args.resume_epoch,args.epochs):
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        print('-' * 5)
        for phase in ['train','val']: 
            if args.arch in model_names:
                if phase == 'train':
                    for i,p in enumerate(PRIMES):  
                        scheduler[i].step()
                        model[i].train()
                else:
                    for i,p in enumerate(PRIMES):    
                        model[i].eval()
            elif args.arch in rnet.__dict__:
                if phase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()
        
            num=0 
            csum=0       
            running_loss=0.0
            cur=0
            cur_loss=0.0

    
            print(phase,':')
            end = time.time()
            for ib, data in enumerate(dataloader[phase]):
                data_time=time.time() - end
                inputs = data[0]["data"].to(device, non_blocking=True)
                targets= data[0]["label"].squeeze().to(device, non_blocking=True)
                if args.arch in model_names:
                    for i,p in enumerate(PRIMES):
                            optimizer[i].zero_grad()
                elif args.arch in rnet.__dict__:
                    optimizer.zero_grad()
                    
                batch_size = targets.size(0)
                correct=torch.ones((batch_size),dtype=torch.uint8).to(device)
                with torch.set_grad_enabled(phase == 'train'):
                    if args.arch in model_names:
                        for i,p in enumerate(PRIMES):
                            outputs=model[i](inputs)
                            targetp=(targets%p).long()
                            loss = criterion(outputs,targetp)
                            if phase == 'train':
                                #loader_len = int(dataloader[phase]._size / args.batch_size)
                                #adjust_learning_rate(optimizer[i], epoch,ib+1, loader_len)
                                if args.fp16:
                                    optimizer[i].backward(loss)
                                else:
                                    loss.backward()
                                optimizer[i].step()
                            _, pred = outputs.topk(1, 1, True, True)
                            correct = correct.mul(pred.view(-1).eq(targetp))
                    elif args.arch in rnet.__dict__:
                        outputs=model(inputs)
                        loss=0.0
                        for i,p in enumerate(PRIMES):
                            targetp=(targets%p).long()
                            loss += criterion(outputs[i],targetp)
                            _, pred = outputs[i].topk(1, 1, True, True)
                            correct = correct.mul(pred.view(-1).eq(targetp))
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            
                num+=batch_size
                csum+=correct.float().sum(0)
                acc1= csum/num*100
                running_loss += loss.item() * batch_size
                average_loss=running_loss/num
                cur+=batch_size
                cur_loss+=loss.item() * batch_size
                cur_avg_loss=cur_loss/cur
                batch_time=time.time() - end
                end=time.time()
                if (ib+1) % args.print_freq ==0:
                    print('{} L:{:.4f} correct:{:.0f} acc1:{:.4f} data:{:.2f}s batch:{:.2f}s'
                          .format(num,cur_avg_loss,csum,acc1,
                                  data_time,batch_time))
                    cur=0
                    cur_loss=0.0
        
            print('------SUMMARY:',phase,'---------')
            print('E:{} L:{:.4f} correct:{:.0f} acc1: {:.4f} Time: {:.4f}s'
                      .format(epoch,average_loss,csum,acc1,batch_time))
            dataloader[phase].reset()
        
        '''save the state'''
        save_file=os.path.join(args.save_folder,'epoch_'+str(epoch+1)+'.pth')
        save_dict={'epoch':epoch + 1,
                   'acc':acc1,
                   'arch':args.arch,
                   }
        if args.arch in model_names:
            for i,p in enumerate(PRIMES):
                save_dict['state_'+str(p)]=model[i].state_dict()
                save_dict['optim_'+str(p)]=optimizer[i].state_dict()
        elif args.arch in rnet.__dict__:
            save_dict['state']=model.state_dict()
            save_dict['optim']=optimizer.state_dict()
            save_dict['primes']=PRIMES
        torch.save(save_dict,save_file)
        if acc1>best_acc:
            shutil.copyfile(save_file, 'model_best.pth.tar')
                
if __name__ == '__main__':
    main()   
    
    