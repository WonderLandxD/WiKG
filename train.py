import sys
import os
import argparse
import csv
import numpy as np
from model import WiKG
import torch
import torch.nn.functional as F
import torch.utils.data as data
import torch.backends.cudnn as cudnn

from pathlib import Path

from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix

import pandas as pd


class esca_2type_Datasets(data.Dataset):
    def __init__(self, utils_dir, fold, data_type='train', device='cuda:0'):
        fold_path = os.path.join(utils_dir, f'fold_{fold}')
        csv_file = os.path.join(fold_path, f'{data_type}_data.csv')
        pd_data = pd.read_csv(csv_file)
        self.data_list = pd_data['filepath'].tolist()
        self.device = device

    def __getitem__(self, item):
        data_path = self.data_list[item]
        data = torch.load(data_path, map_location=self.device)
        label_name = data_path.split('/')[-4]
        if label_name == 'adenomas':
            label = 0
        elif label_name == 'squamous':
            label = 1
        else:
            ValueError
        return data, label
    
    def __len__(self):
        return len(self.data_list)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = torch.zeros(1).to(device)
    train_loader = tqdm(train_loader, file=sys.stdout, ncols=100, colour='red')

    for i, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)
        label = label.to(device)
        logits = model(data)

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()

        total_loss = (total_loss * i + loss.detach()) / (i + 1)
        train_loader.desc = 'Train\t[epoch {}] lr: {}\tloss {}'.format(epoch, optimizer.param_groups[0]["lr"], round(total_loss.item(), 3))

    return logits


@torch.no_grad()
def val_one_epoch(model, val_loader, device, data_type='val'):
    model.eval()
    labels = torch.tensor([], device=device)
    preds = torch.tensor([], device=device)
    if data_type == 'val':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=100, colour='blue')
    elif data_type == 'test':
        val_loader = tqdm(val_loader, file=sys.stdout, ncols=100, colour='green')

    for i, (data, label) in enumerate(val_loader):
        data = data.to(device)
        label = label.to(device)
        output = model(data)
        labels = torch.cat([labels, label], dim=0)
        preds = torch.cat([preds, output.detach()], dim=0)

    return preds.cpu(), labels.cpu()


def cal_metrics(logits, labels, num_classes):       # logits:[batch_size, num_classes]   labels:[batch_size, ]
    # accuracy
    predicted_classes = torch.argmax(logits, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())

    # macro-average area under the cureve (AUC) scores
    probs = F.softmax(logits, dim=1)
    if num_classes > 2:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
    else:
        auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())

    # weighted f1-score
    f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')

    # quadratic weighted Kappa
    kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')

    # macro specificity 
    specificity_list = []
    for class_idx in range(num_classes):
        true_positive = np.sum((labels.numpy() == class_idx) & (predicted_classes.numpy() == class_idx))
        true_negative = np.sum((labels.numpy() != class_idx) & (predicted_classes.numpy() != class_idx))
        false_positive = np.sum((labels.numpy() != class_idx) & (predicted_classes.numpy() == class_idx))
        false_negative = np.sum((labels.numpy() == class_idx) & (predicted_classes.numpy() != class_idx))

        specificity = true_negative / (true_negative + false_positive)
        specificity_list.append(specificity)

    macro_specificity = np.mean(specificity_list)

    # confusion matrix
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())

    return accuracy, auc, f1, kappa, macro_specificity, confusion_mat



def parse():
    parser = argparse.ArgumentParser('Training for WiKG')
    parser.add_argument('--epochs', type=int, default=100)
    
    parser.add_argument('--embed_dim', type=int, default=384, help="The dimension of instance-level representations")
    
    parser.add_argument('--utils', type=str, default=None, help='utils path')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use for training / testing')
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--seed', default=2023, type=int)  # 3407, 1234, 2023
    parser.add_argument('--n_classes', type=int, default=2)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--save_dir', default=None, help='path where to save')
    parser.add_argument('--encoder_name', default='vitsmall', help='fixed encoder name, for saving folder name')

    return parser.parse_args()


def main(args):
    print(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.benchmark = True

    utils_dir = args.utils

    train_set = esca_2type_Datasets(utils_dir=utils_dir, fold=args.fold, data_type='train', adj_ok=False, device='cuda:0')
    val_set = esca_2type_Datasets(utils_dir=utils_dir, fold=args.fold, data_type='val', adj_ok=False, device='cuda:0')

    print(f'Using fold {args.fold}')
    print(f'train: {len(train_set)}')
    print(f'valid: {len(val_set)}')

    train_loader = data.DataLoader(train_set, batch_size=1, num_workers=args.n_workers, shuffle=True)
    val_loader = data.DataLoader(val_set, batch_size=1, num_workers=args.n_workers, shuffle=False)

    model = WiKG(dim_in=args.embed_dim, dim_hidden=512, topk=6, n_classes=args.n_classes, agg_type='bi-interaction', dropout=0.3, pool='mean').to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    criterion = torch.nn.CrossEntropyLoss()
    
    output_dir = Path(os.path.join(args.save_dir, args.encoder_name, f'fold{args.fold}'))
    weight_dir = output_dir / "weight"

    os.makedirs(output_dir, exist_ok=True)

    os.makedirs(weight_dir, exist_ok=True)

    print(f"Start training for {args.epochs} epochs")

    with open(f'{output_dir}/results.csv', 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['epoch', 'val acc', 'val auc', 'val f1', 'val kappa', 'val specificity'])

    with open(f'{output_dir}/val_matrix.txt', 'w') as f:
            print('test start', file=f)

    max_val_accuracy = 0.0
    max_val_auc = 0.0
    max_val_f1 = 0.0
    max_val_kappa = 0.0
    max_val_specificity = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        train_logits = train_one_epoch(model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device, epoch=epoch + 1)

        val_preds, val_labels = val_one_epoch(model=model, val_loader=val_loader, device=device, data_type='val')
        val_acc, val_auc, val_f1, val_kappa, val_specificity, val_mat = cal_metrics(val_preds, val_labels, num_classes=args.n_classes)
        print('Val\t[epoch {}] acc:{}\tauc:{}\tf1-score:{}\tkappa:{}\tspecificity:{}'.format(epoch + 1, val_acc, val_auc, val_f1, val_kappa, val_specificity))
        print('val matrix ......')
        print(val_mat)
        
        max_val_accuracy = max(max_val_accuracy, val_acc)
        max_val_auc = max(max_val_auc, val_auc)
        max_val_f1 = max(max_val_f1, val_f1)
        max_val_kappa = max(max_val_kappa, val_kappa)
        max_val_specificity = max(max_val_specificity, val_specificity)

        if max_val_accuracy == val_acc:
            print('best val acc found... save best acc weights...')
            torch.save({'model': model.state_dict()}, f"{weight_dir}/best_acc.pth")
        
        if max_val_auc == val_auc:
            print('best val auc found... save best auc weights...')
            torch.save({'model': model.state_dict()}, f"{weight_dir}/best_auc.pth")

        if max_val_f1 == val_f1:
            print('best val f1 found... save best f1 weights...')
            torch.save({'model': model.state_dict(),}, f"{weight_dir}/best_f1.pth")

        if max_val_kappa == val_kappa:
            print('best val kappa found... save best kappa weights...')
            torch.save({'model': model.state_dict()}, f"{weight_dir}/best_kappa.pth")

        if max_val_specificity == val_specificity:
            print('best val specificity found... save best specificity weights...')
            torch.save({'model': model.state_dict()}, f"{weight_dir}/best_specificity.pth")

        print('Max val accuracy: {:.4f}%'.format(max_val_accuracy))
        print('Max val auc: {:.4f}%'.format(max_val_auc))
        print('Max val f1: {:.4f}%'.format(max_val_f1))
        print('Max val kappa: {:.4f}%'.format(max_val_kappa))
        print('Max val specificity: {:.4f}%'.format(max_val_specificity))
        with open(f'{output_dir}/val_matrix.txt', 'a') as f:
            print(epoch + 1, file=f)
            print(val_mat, file=f)

        with open(f'{output_dir}/results.csv', 'a') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch+1, val_acc, val_auc, val_f1, val_kappa, val_specificity])
            

if __name__ == '__main__':
    opt = parse()
    main(opt)