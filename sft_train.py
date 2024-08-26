import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import logging
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model import Transformer, ModelArgs
from dataset_sft import SFTDataset


logging.basicConfig(filename='app_test.log', 
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
    
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)

    model.load_state_dict(torch.load("/mnt/data-nas/cy/code/study/baby-llama2-chinese/out/pretrain/epoch_0.pth", map_location="cpu"))
    return model


def train_epoch(epoch):
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        X = X.cuda()
        Y = Y.cuda()

        optimizer.zero_grad()

        logits, targets = model(X, Y)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=0, reduce=False)
        loss_mask = loss_mask.view(-1).cuda()
        loss = torch.sum(loss*loss_mask)/loss_mask.sum()

        loss.backward()
        optimizer.step()

        global_step=epoch*batch_size + step
        logging.info(f"step:{global_step} loss:{loss}")

    checkpoint = {"model_state_dict": model.module.state_dict(), \
                            "optimizer_state_dict": optimizer.state_dict(), \
                            "epoch": epoch}

    torch.save(checkpoint, "./checkpoints/model_sft_{}.pth".format(epoch))


if __name__ == "__main__":

    logging.info('model sft start')

    batch_size = 128
    epoch = 10
    LR_INIT = 1e-5

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

    data_path_list= pd.read_csv('./sft_data/sft_data.csv')
    train_ds = SFTDataset(data_path_list, max_length=512)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=False,        
        num_workers=0,
    )


    for i in range(epoch):
        model.train()
        train_epoch(epoch)