import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7,6,5,4,3,2,1,0'
import logging
import deepspeed
import argparse
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from model import Transformer, ModelArgs
from dataset import PretrainDataset
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter


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

        model.backward(loss)
        model.step()


        if (args.local_rank == 0):
            global_step=epoch*batch_size + step
            writer.add_scalar("loss_lr/ce_loss", loss, global_step=global_step)
            logging.info(f"step:{global_step} loss:{loss}")
    
    if (args.local_rank == 0):

        checkpoint = {"model_state_dict": model.module.state_dict(), \
                                "optimizer_state_dict": optimizer.state_dict(), \
                                "epoch": epoch}

        torch.save(checkpoint, "./checkpoints/model_{}_{}.pth".format(epoch, step))


def add_argument():
    parser = argparse.ArgumentParser(description='DeepSpeed')
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')
    parser = deepspeed.add_config_arguments(parser) # 参数传入deepspeed
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    logging.info('model start')

    batch_size = 96
    epoch = 2
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


    args = add_argument()
    device = torch.device('cuda', args.local_rank)


    if args.local_rank == 0:
        test_n = 1
        log_path = "./logs/log_"+str(test_n) + "/"
        if os.path.exists(log_path):
            os.system("rm -rf {}".format(log_path))
        os.makedirs("./checkpoints/"+str(test_n), exist_ok=True)
        writer=SummaryWriter(log_dir=log_path)


    model = init_model()
    optimizer = optim.AdamW(model.parameters(), lr=LR_INIT, betas=(0.9, 0.98), eps=1.0e-6)

    model, *_ = deepspeed.initialize(args=args, model=model, model_parameters=model.parameters(), optimizer=optimizer)
    scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

    data_path_list= [
                        '/mnt/data-nas/chenyang/code/baby_llama/baby-llama2-chinese/data/pretrain_data.bin'
                    ]

    train_ds = PretrainDataset(data_path_list, max_length=max_seq_len,memmap=True)
    train_sampler = DistributedSampler(train_ds, shuffle=True, drop_last=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=16, sampler=train_sampler)


    for i in range(epoch):
        model.train()
        scheduler.step()
        train_epoch(epoch)

        # model.eval()