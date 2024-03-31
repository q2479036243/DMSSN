import argparse
import warnings
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import sys
sys.path.append("..")
from assistant.dataloader import HL_SC
from models.teacher_model import Auto
from assistant.list import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def parse_args():
    parser = argparse.ArgumentParser(description='DMSSN')
    parser.add_argument('--net', default='teacher', type=str, help='net_name:')
    parser.add_argument('--dataset', default='HL-SOD', type=str, help='dataset:')
    parser.add_argument('--lr', default=0.004, type=float, help='learning rate:')
    parser.add_argument('--st', default=10, type=int, help='step_size:')
    parser.add_argument('--ga', default=0.1, type=float, help='gamma:')
    parser.add_argument('--batchsize', default=4, type=int, help='batchsize:')
    parser.add_argument('--gpus', default=4, type=int, help='number of gpus:')
    parser.add_argument('--nEpochs', default=50, type=int, help='number of epochs:')
    parser.add_argument('--sc_dir', default='/data3/QHL/DMSSN/SC_out/', type=str,
                        help='dataset dir:')
    parser.add_argument('--hlsod_dir', default='/data3/QHL/DATA/SOD/HL-SOD/', type=str,
                        help='dataset dir:')
    parser.add_argument('--netdir', default='/data3/QHL/DMSSN/netdir/', type=str, help='save model dir:')
    parser.add_argument('--pretrained', default=False, type=bool, help='whether or not: pretrained')
    parser.add_argument('--prenet', default='/data3/QHL/HSOD/ACEN/netdir/teacher/teacher_epoch_5.pth', type=str,
                        help='model name:')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    return args


fuse_loss = nn.SmoothL1Loss()
def sam_loss(output,input):
    batchsize = output.shape[0]
    sam_loss = 0
    for b in range(batchsize):
        dot_sum = torch.sum(output[b,:,:,:]*input[b,:,:,:],axis=0)
        norm_out = torch.linalg.norm(output[b,:,:,:], axis=0)
        norm_in = torch.linalg.norm(input[b,:,:,:], axis=0)
        res = torch.arccos(dot_sum/(norm_out*norm_in+1e-7))
        sam_loss += torch.mean(res)
    sam_loss = sam_loss/batchsize
    return sam_loss


warnings.filterwarnings("ignore")
args = parse_args()
print(args)


torch.distributed.init_process_group('nccl')
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)


net_save_path = args.netdir + args.net + '/'
if not os.path.exists(net_save_path):
    os.mkdir(net_save_path)


if args.dataset == "HL-SOD":
    #train_list = glob.glob(args.gmm_dir+'*.mat')
    train_list = all_lists
    image_path = args.sc_dir
    label_path = args.hlsod_dir + 'ground_truth/'
    train_dataset = HL_SC(
        img_list=train_list,
        img_path=image_path,
        lab_path=label_path,
        trans = True)


train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, shuffle=False, sampler=train_sampler, num_workers=1, drop_last=True)
#train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize*args.gpus, shuffle=True, num_workers=0, drop_last=True)


model = Auto()
'''inputs = torch.randn(1, 200, 512, 512)
flops, params = profile(model, (inputs,))
print('flops: ', flops, 'params: ', params)'''
'''if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)'''
model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)


#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.001)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


if args.pretrained:
    checkpoint = torch.load(args.prenet)
    model.load_state_dict(checkpoint, strict=True)


train_min_loss = 100
train_min_epoch = 0

def train(model, train_loader, epoch):
    global train_min_loss
    global train_min_epoch
    if torch.distributed.get_rank() == 0:
        print("\n")
        print("Epoch---",epoch+1)
    model.train()
    train_loss = 0
    len = 0
    for i, data in enumerate(train_loader):
        inputs = data['image']
        inputs_v = inputs.to(device)
        optimizer.zero_grad()
        enc64,enc32,dec64,dec200 = model(inputs_v)
        fuse_out = fuse_loss(dec200, inputs_v)
        sam_out = sam_loss(dec200, inputs_v)
        loss = fuse_out + sam_out
        len = len + 1
        loss.backward()
        optimizer.step()
        train_loss += loss
        if torch.distributed.get_rank() == 0:
            print("Batch: ", len * args.batchsize * args.gpus, "Loss={:.4f}".format(loss.item()))
    loss_mean = train_loss / len
    if loss_mean < train_min_loss:
        train_min_loss = loss_mean
        train_min_epoch = epoch + 1
    if torch.distributed.get_rank() == 0:
        print("Train:")
        print("Loss_mean={:.4f}".format(loss_mean.item()))
        print('Best_Epoch:',train_min_epoch,"Best_Loss:{:.4f}".format(train_min_loss))


def main():
    for epoch in range(0, args.nEpochs):
        train(model, train_dataloader, epoch)
        scheduler.step()
        if ((epoch + 1) % 1 == 0):
            if torch.distributed.get_rank() == 0:
                model_name = args.net + "_epoch_%d.pth" % (epoch + 1)
                torch.save(model.state_dict(), os.path.join(net_save_path, model_name), _use_new_zipfile_serialization=True)


if __name__ == "__main__":
    main()