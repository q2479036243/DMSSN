import argparse
import warnings
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.optim as optim
import thop
import sys
sys.path.append("..")
import assistant.pytorch_ssim
import assistant.pytorch_iou
from assistant.dataloader import HL_SC
from models.DSST_teacher import Auto
from models.DSST_double import ACEN
from assistant.list import *
from assistant.evaluate_function import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"


class Regularization(torch.nn.Module):
    def __init__(self,model,weight_decay,p=1):
        '''
        :param model 模型
        :param weight_decay:正则化参数
        :param p: 当p=2为L2正则化,p=1为L1正则化
        '''
        super(Regularization, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model=model
        self.weight_decay=weight_decay
        self.p=p
        self.weight_list=self.get_weight(model)
        #self.weight_info(self.weight_list)
    def to(self,device):
        '''
        指定运行模式
        :param device: cude or cpu
        :return:
        '''
        self.device=device
        super().to(device)
        return self
    def forward(self, model):
        self.weight_list=self.get_weight(model)#获得最新的权重
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss
    def get_weight(self,model):
        '''
        获得模型的权重列表
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list
    def regularization_loss(self,weight_list, weight_decay, p=1):
        '''
        计算张量范数
        :param weight_list:
        :param p: 范数计算中的幂指数值，默认求2范数
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss=0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg
        reg_loss=weight_decay*reg_loss
        return reg_loss
    def weight_info(self,weight_list):
        '''
        打印权重列表信息
        :param weight_list:
        :return:
        '''
        print("---------------regularization weight---------------")
        for name ,w in weight_list:
            print(name)
        print("---------------------------------------------------")


def parse_args():
    parser = argparse.ArgumentParser(description='HSOD')
    parser.add_argument('--net', default='DMSSN_double', type=str, help='net_name:')
    parser.add_argument('--dataset', default='HL-SOD', type=str, help='dataset:')
    parser.add_argument('--lr', default=0.06, type=float, help='learning rate:')
    parser.add_argument('--st', default=20, type=int, help='step_size:')
    parser.add_argument('--ga', default=0.6, type=float, help='gamma:')
    parser.add_argument('--batchsize', default=6, type=int, help='batchsize:')
    parser.add_argument('--gpus', default=4, type=int, help='number of gpus:')
    parser.add_argument('--nEpochs', default=100, type=int, help='number of epochs:')
    parser.add_argument('--sc_dir', default='/data3/QHL/DMSSN/SC_out/', type=str,
                        help='dataset dir:')
    parser.add_argument('--hlsod_dir', default='/data3/QHL/DATA/SOD/HL-SOD/', type=str,
                        help='dataset dir:')
    parser.add_argument('--netdir', default='/data3/QHL/DMSSN/netdir/', type=str, help='save model dir:')
    parser.add_argument('--pretrained', default=False, type=bool, help='whether or not: pretrained')
    parser.add_argument('--prenet', default='/data3/QHL/DSST/netdir/DSST/train_2_epoch_55.pth', type=str,
                        help='model name:')
    parser.add_argument('--teacher', default='/data3/QHL/DMSSN/netdir/teacher/teacher_epoch_30.pth', type=str,
                        help='model name:')                 
    parser.add_argument('--L1decay', default=0.0001, type=float, help='L1 weight_decay:')
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    return args


bce_loss = nn.BCELoss(size_average=True)
ssim_loss = assistant.pytorch_ssim.SSIM(window_size=11,size_average=True)
iou_loss = assistant.pytorch_iou.IOU(size_average=True)
def sod_loss(pred,target):
    bce_out = bce_loss(pred,target)
    ssim_out = 1 - ssim_loss(pred,target)
    iou_out = iou_loss(pred,target)
    loss = bce_out + ssim_out + iou_out
    return loss


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
def auto_loss(pred,target):
    fuse_out = fuse_loss(pred,target)
    sam_out = sam_loss(pred,target)
    loss = fuse_out + sam_out
    return loss


warnings.filterwarnings("ignore")
args = parse_args()
print(args)


torch.distributed.init_process_group('nccl')
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)


net_save_path = args.netdir + args.net + '/'
if not os.path.exists(net_save_path):
    if torch.distributed.get_rank() == 0:
        os.mkdir(net_save_path)


if args.dataset == "HL-SOD":
    train_list = train_lists
    test_list = test_lists

    image_path = args.sc_dir
    label_path = args.hlsod_dir + 'ground_truth/'
    # label_path = '/data3/QHL/DMSSN/ground_truth/'
    train_dataset = HL_SC(
            img_list=train_list,
            img_path=image_path,
            lab_path=label_path,
            trans = True)
    test_dataset = HL_SC(
            img_list=test_list,
            img_path=image_path,
            lab_path=label_path,
            trans = False)

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=args.batchsize, sampler=train_sampler, num_workers=6, drop_last=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize, sampler=test_sampler, num_workers=6, drop_last=True)

model = ACEN()
x = torch.randn(1,200,512,512)
flops, params = thop.profile(model,inputs=(x,))
print(flops, params)
teacher = Auto()
model.to(device)
teacher.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)
teacher = torch.nn.parallel.DistributedDataParallel(teacher, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=False)

#optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=0.001)
optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.6)
    
if args.L1decay>0:
    reg_loss=Regularization(model, args.L1decay, p=1).to(device)
else:
    print("no L1 regularization")

if args.pretrained:
    checkpoint = torch.load(args.prenet)
    model.load_state_dict(checkpoint, strict=True)
        
check = torch.load(args.teacher)
teacher.load_state_dict(check, strict=False)

train_min_loss = 100
train_min_epoch = 0
test_min_loss = 100
test_min_epoch = 0

def train(model, teacher, train_loader, epoch):
    global train_min_loss
    global train_min_epoch
    if torch.distributed.get_rank() == 0:
        print("\n")
        print("Epoch---",epoch+1)
    train_sampler.set_epoch(epoch)
    test_sampler.set_epoch(epoch)
    model.train()
    teacher.eval()
    train_loss = 0
    len = 0
    for i, data in enumerate(train_loader):
        inputs, labels = data['image'], data['label']
        inputs_v, labels_v = inputs.to(device), labels.to(device)
        with torch.no_grad():
            tea_enc64, tea_enc32, tea_dec64 = teacher(inputs_v)
        optimizer.zero_grad()
        enc64, enc32, dec64, dec200, out = model(inputs_v)
        loss_dit = fuse_loss(enc64, tea_enc64) + fuse_loss(enc32, tea_enc32) + fuse_loss(dec64, tea_dec64)
        loss_auto = auto_loss(dec200, inputs_v)
        loss_sod = sod_loss(out, labels_v)
        loss_reg = reg_loss(model).item()
        loss = loss_dit + loss_sod + loss_reg + loss_auto
        len = len + 1
        loss.backward()
        optimizer.step()
        train_loss += loss
        # print("Batch: ", len * args.batchsize * args.gpus, "Loss={:.4f}".format(loss.item()))
    loss_mean = train_loss / len
    if loss_mean < train_min_loss:
        train_min_loss = loss_mean
        train_min_epoch = epoch + 1
    # if torch.distributed.get_rank() == 0:
    print("Train:")
    print("Loss_mean={:.4f}".format(loss_mean.item()))
    print('Best_Epoch:',train_min_epoch,"Best_Loss:{:.4f}".format(train_min_loss))


def test(model, teacher, test_loader, epoch):
    global test_min_loss
    global test_min_epoch
    model.eval()
    teacher.eval()
    test_loss = 0
    len = 0
    mae, pre, rec, f_1, auc, cc, nss = 0, 0, 0, 0, 0, 0, 0
    mini_batch = args.batchsize
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, name = data['image'], data['label'], data['id']
            inputs_v, labels_v = inputs.to(device), labels.to(device)
            enc64, enc32, dec64, dec200, out = model(inputs_v)
            tea_enc64, tea_enc32, tea_dec64 = teacher(inputs_v)
            loss_dit = fuse_loss(enc64, tea_enc64) + fuse_loss(enc32, tea_enc32) + fuse_loss(dec64, tea_dec64)
            loss_auto = auto_loss(dec200, inputs_v)
            loss_sod = sod_loss(out, labels_v)
            loss_reg = reg_loss(model).item()
            loss = loss_dit + loss_sod + loss_reg + loss_auto
            len = len + 1
            for j in range(mini_batch):
                mask, label = out[j,:,:,:], labels_v[j,:,:,:]
                mask = mask.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                try:
                    mae_, pre_, rec_, f_1_, auc_, cc_, nss_ = evaluate(mask, label)
                    mae = mae + mae_
                    pre = pre + pre_
                    rec = rec + rec_
                    f_1 = f_1 + f_1_
                    auc = auc + auc_
                    cc = cc + cc_
                    nss = nss + nss_
                except:
                    print(name[j])
            test_loss += loss
        loss_mean = test_loss / len
        len = len * mini_batch
        mae = mae / len
        pre = pre / len
        rec = rec / len
        f_1 = f_1 / len
        auc = auc / len
        cc = cc / len
        nss = nss / len
        if loss_mean < test_min_loss:
            test_min_loss = loss_mean
            test_min_epoch = epoch + 1
        # if torch.distributed.get_rank() == 0:
        print("Test:")
        print("Loss_mean={:.4f}".format(loss_mean.item()),"Auc={:.4f}".format(auc))
        print("Pre={:.4f}".format(pre),"Rec={:.4f}".format(rec),"F_1={:.4f}".format(f_1))
        print("CC={:.4f}".format(cc),"Nss={:.4f}".format(nss),"Mae={:.4f}".format(mae))
        print('Best_Epoch:',test_min_epoch,"Best_Loss:{:.4f}".format(test_min_loss))


def main():
    for epoch in range(0, args.nEpochs):
        train(model, teacher, train_dataloader, epoch)
        test(model, teacher, test_dataloader, epoch)
        scheduler.step()
        if ((epoch + 1) % 5 == 0):
            if torch.distributed.get_rank() == 0:
                model_name = args.net + "_epoch_%d.pth" % (epoch + 1)
                torch.save(model.state_dict(), os.path.join(net_save_path, model_name), _use_new_zipfile_serialization=True)


if __name__ == "__main__":
    main()