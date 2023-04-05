import torch
import torch.nn as nn
import numpy as np
import argparse
import tqdm
import os
from torch.utils import tensorboard
from utils import create_dataloader
import torch.optim.lr_scheduler as lr_scheduler
from speechbrain.nnet.normalization import PCEN

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
    #if isinstance(m, nn.Conv2d):
        #torch.nn.init.xavier_uniform_(m.weight)
    #if isinstance(m, nn.Conv3d):
        #torch.nn.init.xavier_uniform_(m.weight)


class AudioModel(nn.Module):
    def __init__(self):
        super().__init__()

       # self.pcen = PCEN(147)
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5,5), padding='valid', stride=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_1 = nn.BatchNorm2d(num_features=4)
        self.relu = nn.ReLU()

        self.conv_block_1 = nn.Sequential( self.conv_1, self.pool_1, self.bn_1, self.relu)
        
        self.conv_2 = nn.Conv2d(in_channels=4,out_channels=2,kernel_size=(3,3), padding='valid', stride=1)
        self.pool_2= nn.MaxPool2d(kernel_size=3, stride=2)
        self.bn_2 = nn.BatchNorm2d(num_features=2)

        self.conv_block_2 = nn.Sequential(self.conv_2, self.pool_2, self.bn_2, self.relu)

        self.conv_3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=(3,3), padding='valid')
        self.pool_3= nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn_3 = nn.BatchNorm2d(num_features=1)

        self.conv_block_3 = nn.Sequential(self.conv_3, self.pool_3, self.relu)

        self.fc_1 = nn.Linear(in_features=864, out_features=64)
        self.fc_2 = nn.Linear(in_features=64, out_features=2)

        self.linear_layer = nn.Sequential(self.fc_1, self.relu, self.fc_2)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.conv_3(x)
        x = x.reshape(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = self.linear_layer(x)
        return x

def model_checkpoint(model, epoch, optimizer, lr_sched, log_dir):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'lr_sched': lr_sched
    }
    torch.save(checkpoint, f'{log_dir}/checkpoint/checkpoint_{epoch}.pth')


def model_training(args, writer, log_dir):
    progress_bar = tqdm.tqdm(range(args.epochs))
    training_dataloader, val_dataloader = create_dataloader(args.batch_size)
    model = AudioModel()
    model.to(DEVICE)
    model.apply(initialize_weights)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
    criterion = nn.BCEWithLogitsLoss(weight=torch.tensor([0, args.weight]))

    for epoch in progress_bar:

        step = 0
        train_loss = []
        for X_train, y_train in training_dataloader:
            X_train.to(DEVICE)
            y_train.to(DEVICE)
            X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1], X_train.shape[2])
            y_train = nn.functional.one_hot(y_train, num_classes=2).view(y_train.shape[0], 2).float()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            iteration = epoch*(len(training_dataloader)) + step
            train_loss.append(loss.detach().cpu().item())
            step +=1
        scheduler.step()
        
    
        #at the end of each epoch, run validation set
        with torch.inference_mode():
            val_loss = []
            for X_val, y_val in val_dataloader:
                X_val.to(DEVICE)
                y_val.to(DEVICE)

                X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1], X_val.shape[2])
                y_val= nn.functional.one_hot(y_val, num_classes=2).view(y_val.shape[0], 2).float()

                logits = model(X_val)
                loss = criterion(logits, y_val)
                val_loss.append(loss.detach().cpu().item())

        writer.add_scalars(f'loss', {
            'train_loss': sum(train_loss)/len(train_loss),
            'val_loss': sum(val_loss)/ len(val_loss)
        }, iteration)

        print(f'epoch {epoch}, train_loss: {sum(train_loss) / len(train_loss)}, val_loss: {sum(val_loss)/ len(val_loss)}')

        #checkpoint the model
        model_checkpoint(model,epoch, optimizer, scheduler, log_dir)


def main(args):
    log_dir = args.log_dir
    if log_dir is None:
        log_dir = f'./logs/audiomodel.batch_size:{args.batch_size}.lr:{args.lr}.lr_schedule:{args.lr_schedule}.weight_decay:{args.weight_decay}.weight:{args.weight}.epochs:{args.epochs}'

    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
        os.mkdir(log_dir + "/checkpoint")

    writer = tensorboard.SummaryWriter(log_dir=log_dir)
    model_training(args, writer, log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parse user input arguments into function')
    parser.add_argument('--epochs',type=int, default=42, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='number of trianing examples in training batch')
    parser.add_argument('--lr', type=int, default=.0005, help='sample rate of recording')
    parser.add_argument('--weight', type=float, default=4.0, help='number of positive label samples that will be recorded')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='number of positive label samples that will be recorded')
    parser.add_argument('--lr_schedule', type=int, default=5, help='learning rate schedule (gamma decay)')
    parser.add_argument('--log_dir', type=int, default=None, help='number of channels that the recorded audio will contain')
    args = parser.parse_args()
    main(args)