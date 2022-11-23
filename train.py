import torch
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import random
import logging
import tqdm
import time

from data.wsi import WSI
from model.gcn import GCN
from torch.utils.data import DataLoader

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3'


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_logging(args):
    if not os.path.exists("./output/log/"): 
        os.makedirs("./output/log/")
    filename = args.log_file_path
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


def parse_args():
    parser = argparse.ArgumentParser(description='GCN')

    # dataset config
    parser.add_argument('--log_file_path', type=str, default="./output/log/xxxx.txt")
    parser.add_argument('--model_name', type=str, default="XXXX")
    parser.add_argument('--ckpt_path', type=str, default="./output/checkpoint/")

    # model train
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--T_max', type=int, default=40)
    parser.add_argument('--eta_min', type=float, default=0.)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--val_freq', type=int, default=1)

    # model config
    parser.add_argument('--dim', type=int, default=64)

    # dataset
    parser.add_argument('--train_label_file_path', type=str, default="./data/name_list.txt")
    parser.add_argument('--valid_label_file_path', type=str, default="./data/name_list.txt")
    parser.add_argument('--train_dataset_path', type=str, default="")
    parser.add_argument('--valid_dataset_path', type=str, default="")
    args = parser.parse_args()
    return args


def run():
    setup_seed(20)
    args = parse_args()

    set_logging(args)

    TrainDataset = WSI(label_file_path=args.train_label_file_path, dataset_path=args.train_dataset_path)
    ValidDataset = WSI(label_file_path=args.valid_label_file_path, dataset_path=args.valid_dataset_path)

    train_loader = DataLoader(dataset=TrainDataset, batch_size=args.batch_size, 
        num_workers=args.num_workers, drop_last=True, shuffle=True)
    val_loader = DataLoader(dataset=ValidDataset, batch_size=args.batch_size,
        num_workers=args.num_workers, drop_last=True, shuffle=False)
    
    net = GCN(dim=64)
    net = nn.DataParallel(net).cuda()
    logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

    # loss function
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    best_acc = 0
    for epoch in args.num_epochs:
        start_time = time.time()
        logging.info('Running training epoch {}'.format(epoch + 1))

        net.train()
        loss_list = []
        flag_list = []

        for data in tqdm(train_loader):
            image, class_label = data
            image = image.cuda()
            class_label = class_label.cuda()
            class_label = torch.squeeze(class_label).cuda()

            pred_label = net(image)
            pred_label = torch.squeeze(pred_label)

            optimizer.zero_grad()
            loss = criterion(pred_label, class_label)
            loss_list.append(loss.item())

            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_label = pred_label.cpu().detach().numpy()
            class_label = class_label.cpu().detach().numpy()

            for i in range(pred_label.shape[0]):
                flag = 1 if round(pred_label[i]) == class_label[i] else 0
                flag_list.append(flag)

        avg_loss = loss_list / len(loss_list)
        acc = sum(flag_list) / len(flag_list)
        logging.info('--Training ACC: {:.2f}; Loss: {:.4f}'.format(acc, avg_loss))

        if (epoch + 1) % args.val_freq == 0:
            logging.info('Starting eval...')
            logging.info('Running val in epoch {}'.format(epoch + 1))
            with torch.no_grad():
                net.eval()
                flag_list = []
                for data in tqdm(val_loader):
                    image, class_label = data
                    image = image.cuda()
                    class_label = class_label.cuda()
                    class_label = torch.squeeze(class_label).cuda()

                    pred_label = net(image)
                    pred_label = torch.squeeze(pred_label)

                    pred_label = pred_label.cpu().detach().numpy()
                    class_label = class_label.cpu().detach().numpy()

                    for i in range(pred_label.shape[0]):
                        flag = 1 if round(pred_label[i]) == class_label[i] else 0
                        flag_list.append(flag)
                
                acc = sum(flag_list) / len(flag_list)
                if acc > best_acc:
                    if not os.path.exists(args.ckpt_path):
                        os.makedirs(args.ckpt_path)
                    best_acc = acc
                    logging.info('======================================================================================')
                    logging.info('======================================================================================')
                    logging.info('best acc is {}'.format(best_acc))

                    # save weights
                    model_name = "epoch{}.pt".format(epoch + 1)
                    model_save_path = os.path.join(args.ckpt_path, model_name)
                    torch.save(net.module.state_dict(), model_save_path)
                    logging.info('Saving weights and model of epoch{}, ACC:{}'.format(epoch + 1, best_acc))
        
        logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))


if __name__ == "__main__":
    cpu_num = 1
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)
    run()