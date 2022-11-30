import os
import torch
import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import logging
import time
import torch.distributed as dist
import torch.multiprocessing as mp

from data.wsi import WSIData
from model.gcn import GCN
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP



def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_log(args):
    if not os.path.exists(args.log_file_path): 
        os.makedirs(args.log_file_path)

    filename = os.path.join(args.log_file_path, args.model_name + ".log")
    log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s")

    writer = logging.getLogger()
    fileHandler = logging.FileHandler(filename)
    fileHandler.setFormatter(log_format)
    writer.addHandler(fileHandler)
    writer.setLevel(logging.INFO)
    return writer


def parse_args():
    parser = argparse.ArgumentParser(description='GCN')

    # dataset config
    parser.add_argument('--log_file_path', type=str, default="./output/log/base_model")
    parser.add_argument('--model_name', type=str, default="baseline-temp")
    parser.add_argument('--ckpt_path', type=str, default="./output/checkpoint/")

    # model train
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--T_max', type=int, default=40)
    parser.add_argument('--eta_min', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--val_freq', type=int, default=1)

    # model config
    parser.add_argument('--input_dim', type=int, default=3)
    parser.add_argument('--num_class', type=int, default=9)
    parser.add_argument('--dim', type=int, default=32)

    # dataset
    parser.add_argument('--train_dataset_path', type=str, default="/mnt/cpath2/lf/data/wsi_44_demo/train")
    parser.add_argument('--valid_dataset_path', type=str, default="/mnt/cpath2/lf/data/wsi_44_demo/val")
    args = parser.parse_args()
    return args


def main(local_rank, args, train_dataset, valid_dataset):
    args.local_rank = local_rank
    writer = set_log(args)
    args.ckpt_path = os.path.join(args.ckpt_path, args.model_name)

    if not os.path.exists(args.ckpt_path): 
        os.makedirs(args.ckpt_path)

    dist.init_process_group("nccl", rank=args.local_rank, world_size=args.n_gpu)
    torch.cuda.set_device(args.local_rank)

    setup_seed(20)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, drop_last=True, shuffle=False, sampler=train_sampler)
    val_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, drop_last=True, shuffle=False)

    net = GCN(num_class=args.num_class).cuda()
    net = DDP(net, device_ids=[args.local_rank])
    
    if args.local_rank in [0, -1]:
        writer.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))
        writer.info('length train dataset {}'.format(len(train_dataset)))
        writer.info('length val dataset {}'.format(len(valid_dataset)))

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    best_acc = 0
    for epoch in range(args.num_epochs):
        start_time = time.time()

        train_sampler.set_epoch(epoch)
        if args.local_rank in [0, -1]:
            writer.info('Running training epoch {}'.format(epoch + 1))

        net.train()
        loss_list = []
        correct_train = 0

        for data in tqdm(train_loader, position=0):
            image, class_label = data
            image = image.cuda(non_blocking=True)
            class_label = class_label.cuda(non_blocking=True)
            class_label = torch.squeeze(class_label)

            pred_label = net(image)
            
            optimizer.zero_grad()
            loss = criterion(pred_label, class_label)
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_label = pred_label.cpu()
            class_label = class_label.cpu()
            pred = pred_label.data.max(1, keepdim=True)[1]
            correct_train += pred.eq(class_label.data.view_as(pred)).sum()

        avg_loss = sum(loss_list) / len(loss_list)
        acc_train = correct_train / (len(train_loader.dataset) / args.n_gpu)

        if args.local_rank in [0, -1]:
            writer.info('Training Loss: {:.8f}, Training Acc: {:.4f}'.format(avg_loss, acc_train))
            writer.info('Starting eval...')
            writer.info('Running val in epoch {}'.format(epoch + 1))

        with torch.no_grad():
            net.eval()
            correct_val = 0

            for data in tqdm(val_loader):
                image, class_label = data
                image = image.cuda(non_blocking=True)
                class_label = class_label.cuda(non_blocking=True)
                class_label = torch.squeeze(class_label)

                pred_label = net(image)
                pred_label = pred_label.cpu()
                class_label = class_label.cpu()

                pred = pred_label.data.max(1, keepdim=True)[1]
                correct_val += pred.eq(class_label.data.view_as(pred)).sum()
            
            acc_val = correct_val / len(val_loader.dataset)
            if acc_val > best_acc:
                if not os.path.exists(args.ckpt_path):
                    os.makedirs(args.ckpt_path)
                best_acc = acc_val
                if args.local_rank in [0, -1]:
                    writer.info('best acc is ============================{:.4f}'.format(best_acc))

                # save weights
                model_name = "epoch{}.pt".format(epoch + 1)
                model_save_path = os.path.join(args.ckpt_path, model_name)
                
                torch.save(net.module.state_dict(), model_save_path)
                if args.local_rank in [0, -1]:
                    writer.info('Saving weights and model of epoch{}, ACC:{:.4f}'.format(epoch + 1, best_acc))
            
            else:
                if args.local_rank in [0, -1]:
                    writer.info('Valid acc is {:.4f}'.format(acc_val))
        if args.local_rank in [0, -1]:
            writer.info('Epoch {} done. Time: {:.2} min'.format(epoch + 1, (time.time() - start_time) / 60))


if __name__ == "__main__":
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    args = parse_args()

    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "18101"

    args.n_gpu = torch.cuda.device_count()
    train_dataset = WSIData(dataset_path=args.train_dataset_path)
    valid_dataset = WSIData(dataset_path=args.valid_dataset_path)
    mp.spawn(main, args=(args, train_dataset, valid_dataset), nprocs=args.n_gpu, join=True)