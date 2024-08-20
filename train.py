import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import time
import math
import pickle
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Transformer, ModelArgs
from dataset import PretrainDataset
from torch.utils.data import Dataset, DataLoader


logging.basicConfig(filename='app.log', 
                    filemode='w',  # 覆盖写入 'a' 表示追加写入
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')


def init_model():
    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=64793,
        multiple_of=multiple_of,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )  
    
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)

    # /mnt/data-nas/cy/code/study/baby-llama2-chinese/out/pretrain/epoch_0.pth
    # model.load_state_dict(torch.load("epoch_0.pth", map_location="cpu"))
    return model


def train_epoch(epoch):
    
    for step, (X, Y) in enumerate(train_loader):
        X = X.cuda()
        Y = Y.cuda()

        optimizer.zero_grad()
        logits, targets = model(X, Y)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        loss.backward()
        optimizer.step()

        logging.info(f"step:{step} loss:{loss} \n")



if __name__ == "__main__":

    logging.info('model start')

    batch_size = 2
    epoch = 1
    LR_INIT = 5e-5


    dim = 512
    n_layers = 8
    n_heads = 8
    multiple_of = 32
    max_seq_len = 512
    dropout = 0.0


    learning_rate = 3e-4 # max learning rate
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95


    model = init_model()
    model = nn.DataParallel(model).cuda()
    optimizer = optim.AdamW(model.parameters(), lr=LR_INIT, betas=(0.9, 0.98), eps=1.0e-6)

    data_path_list= [
                        '/mnt/data-nas/chenyang/code/baby_llama/baby-llama2-chinese/data/pretrain_data.bin'
                    ]

    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len,memmap=True)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=False,
        shuffle=True,        
        num_workers=0
    )


    for i in range(epoch):
        model.train()
        train_epoch(epoch)

        model.eval()
        #