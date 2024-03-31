import argparse
import warnings
import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import utils
import glob
import sys
sys.path.append("..")
from assistant.dataloader import HL_SC
from models.DSST import ACEN
from assistant.list import *
from assistant.evaluate_function import *
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

def parse_args():
    parser = argparse.ArgumentParser(description='DMSSN')
    parser.add_argument('--net', default='DMSSN', type=str, help='net_name:')
    parser.add_argument('--dataset', default='HL-SOD', type=str, help='dataset:')
    parser.add_argument('--batchsize', default=1, type=int, help='batchsize:')
    parser.add_argument('--gpus', default=1, type=int, help='number of gpus:')
    parser.add_argument('--sc_dir_hlsod', default='/data3/QHL/DSST/SC_out/', type=str,
                        help='dataset dir:')
    parser.add_argument('--hlsod_dir', default='/data3/QHL/DATA/SOD/HL-SOD/', type=str,
                        help='dataset dir:')
    parser.add_argument('--output', default='/data3/QHL/DSST/results/DSST/HLSOD/', type=str, help='save result dir')
    parser.add_argument('--pretrained', default=True, type=bool, help='whether or not: pretrained')
    parser.add_argument('--prenet', default='/data3/QHL/DSST/netdir/DSST/DSST_epoch_150.pth', type=str,
                        help='model name:')         
    parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
    args = parser.parse_args()
    return args


warnings.filterwarnings("ignore")
args = parse_args()
print(args)


torch.distributed.init_process_group('nccl')
torch.cuda.set_device(args.local_rank)
device = torch.device("cuda", args.local_rank)

if not os.path.exists(args.output):
    os.mkdir(args.output)
dir_output = os.path.join(args.output, args.net)
if not os.path.exists(dir_output):
    os.mkdir(dir_output)


#test_list = test_lists_hssod
test_list = glob.glob(args.sc_dir_hlsod + '*' + '.mat')

image_path = args.sc_dir_hlsod
label_path = args.hlsod_dir + 'ground_truth/'
test_dataset = HL_SC(
        img_list=test_list,
        img_path=image_path,
        lab_path=label_path,
        trans = False)

test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=args.batchsize, shuffle=False, sampler=test_sampler, num_workers=0, drop_last=False)

model = ACEN()
model.to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],output_device=args.local_rank,find_unused_parameters=True)


if args.pretrained:
    checkpoint = torch.load(args.prenet)
    model.load_state_dict(checkpoint, strict=True)

def save_image_tensor(input_tensor: torch.Tensor, filename):
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    utils.save_image(input_tensor, filename)


def test(model, test_loader):
    model.eval()
    len = 0
    mae, pre, rec, f_1, auc, cc, nss = 0, 0, 0, 0, 0, 0, 0
    mini_batch = args.batchsize
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels, name = data['image'], data['label'], data['id']
            inputs_v, labels_v = inputs.to(device), labels.to(device)
            enc64, enc32, dec64, dec81, out = model(inputs_v)
            len = len + 1
            for j in range(mini_batch):
                draw, label, id = out[j,:,:,:], labels_v[j,:,:,:], name[j]
                mask = draw.cpu().detach().numpy()
                label = label.cpu().detach().numpy()
                mae_, pre_, rec_, f_1_, auc_, cc_, nss_ = evaluate(mask, label)
                print("\n")
                print(id,":","Pre={:.4f}".format(pre_),"Rec={:.4f}".format(rec_),"F_1={:.4f}".format(f_1_))
                print("Auc={:.4f}".format(auc_),"CC={:.4f}".format(cc_),"Nss={:.4f}".format(nss_),"Mae={:.4f}".format(mae_))
                draw = (draw - torch.min(draw)) / (torch.max(draw) - torch.min(draw))
                draw = torch.swapaxes(draw, 2, 1)
                save_image_tensor(draw, os.path.join(dir_output, name[j] + ".jpg"))
                mae = mae + mae_
                pre = pre + pre_
                rec = rec + rec_
                f_1 = f_1 + f_1_
                auc = auc + auc_
                cc = cc + cc_
                nss = nss + nss_
        len = len * mini_batch
        mae = mae / len
        pre = pre / len
        rec = rec / len
        f_1 = f_1 / len
        auc = auc / len
        cc = cc / len
        nss = nss / len
        print("\n")
        print("Auc={:.4f}".format(auc))
        print("Pre={:.4f}".format(pre),"Rec={:.4f}".format(rec),"F_1={:.4f}".format(f_1))
        print("CC={:.4f}".format(cc),"Nss={:.4f}".format(nss),"Mae={:.4f}".format(mae))


def main():
    test(model, test_dataloader)

if __name__ == "__main__":
    main()