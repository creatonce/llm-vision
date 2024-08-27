import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple
from tqdm import tqdm
import re
import csv
import logging
import requests
from io import BytesIO
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import gradio as gr
import torch
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import Dataset
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer


logging.basicConfig(filename='app_test.log', 
                    filemode='w',  # 覆盖写入 'a' 表示追加写入
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1  # defined later by tokenizer
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 2048
    dropout: float = 0.0

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout

        # use flash attention or a manual implementation?
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            mask = torch.full((1, 1, args.max_seq_len, args.max_seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            self.register_buffer("mask", mask)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        # RoPE relative positional embeddings
        xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # make heads into a batch dimension
        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        # flash implementation
        if self.flash:
            output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
        else:
            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
            assert hasattr(self, 'mask')
            scores = scores + self.mask[:, :, :seqlen, :seqlen]   # (bs, n_local_heads, seqlen, cache_len + seqlen)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)
        return output


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            dropout=args.dropout,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x, freqs_cos, freqs_sin):
        h = x + self.attention.forward(self.attention_norm(x), freqs_cos, freqs_sin)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))
        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor, targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        _bsz, seqlen = tokens.shape
        # import pdb;pdb.set_trace()
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        for layer in self.layers:
            h = layer(h, freqs_cos, freqs_sin)
        h = self.norm(h)

        # if we are given some desired targets also calculate the loss
        logits = self.output(h)

        return logits, targets


def init_model():

    dim = 512
    n_layers = 8
    n_heads = 8
    multiple_of = 32
    max_seq_len = 512
    dropout = 0.0   

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

    # model.load_state_dict(torch.load("/mnt/data-nas/cy/code/study/baby-llama2-chinese/out/pretrain/epoch_0.pth", map_location="cpu"))
    return model


class generate_clip(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.llm = init_model()
        # self.visual = timm.create_model("resnet50", pretrained=True)
        self.visual = models.resnet50(pretrained=False)
        self.visual.fc = nn.Linear(2048, 512)


    def forward(self, text, image):

        text_emb, _ = self.llm(text, text)
        text_emb = text_emb.mean(dim=-1)
        image_emb = self.visual(image)

        return text_emb, image_emb


class get_data(Dataset):
    def __init__(self, split="train", max_txt_length=512):
        super().__init__()
        self.data_root = "/mnt/data-nas/cy/llama-study/cy_llama/llama-vit/data/"
        self.split = split
        self.max_text_length = max_txt_length 
        self.tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')

        resize = 256
        resolution = 224

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(resize),
                transforms.RandomResizedCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(resolution),
                transforms.CenterCrop((resolution,resolution)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }


        self.data = []
        reader = csv.reader(open("/mnt/data-nas/cy/llama-study/cy_llama/llama-vit/data/csv/coca_query_1195694.csv"))
        next(reader)
        for item in tqdm(reader):
            pid, text, image = item
            # if os.path.exists(self.data_root + "images/" + os.path.basename(image)):
            self.data.append([pid, text, self.data_root + "images/" + os.path.basename(image)])


    def convert_to_internal_url(self, url):
            # '''
            # 外网地址转换为内网，下载成功率和下载时间上均有改善，
            # 其中,下载时间由0.0825下降至0.0242
            # '''
            url = url.replace("china-chat-img.soulapp.cn", "soul-chat.oss-cn-hangzhou-internal.aliyuncs.com")
            url = url.replace("china-img.soulapp.cn", "soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
            url = url.replace("img.soulapp.cn", "soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
            url = url.replace("soul-app.oss-cn-hangzhou.aliyuncs.com","soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
            url = url.replace("bratro.soulapp.cn/app", "soul-audit.oss-cn-hangzhou-internal.aliyuncs.com")
            if ("oss-cn-hangzhou-internal.aliyuncs.com" in url):
                url = url.replace("https", "http")
            return url


    def __len__(self):
        max_length = len(self.data)
        print("all data num:{}".format(max_length))
        return max_length


    def __getitem__(self, index):
        sample = self.data[index]
        pid, text, image_path = sample

        pattern = r'(\[.*?\]|<innerTag>.*?</innerTag>)'
        text = re.sub(pattern, '', text)
        text_id = self.tokenizer.encode(text,add_special_tokens=False)[:self.max_text_length]
        text_id = text_id + (self.max_text_length-len(text_id)) * [(self.tokenizer.special_tokens['<eos>'])]
        text_id = torch.LongTensor(text_id)

        try:
            # response = requests.get(self.convert_to_internal_url(image_path))
            # image = Image.open(BytesIO(response.content)).convert('RGB')

            image = Image.open(image_path).convert('RGB')
            image = self.data_transforms[self.split](image)
            return text_id, image
        
        except:
            return None, None

    
    @staticmethod
    def collate_fn(batch):

        batch = [i for i in batch if i[1] is not None]
        text, image = list(zip(*batch))
        text = torch.stack(text, dim=0)
        image = torch.stack(image, dim=0)
        return text, image
    

def convert_to_internal_url(url):
    # '''
    # 外网地址转换为内网，下载成功率和下载时间上均有改善，
    # 其中,下载时间由0.0825下降至0.0242
    # '''
    url = url.replace("china-chat-img.soulapp.cn", "soul-chat.oss-cn-hangzhou-internal.aliyuncs.com")
    url = url.replace("china-img.soulapp.cn", "soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
    url = url.replace("img.soulapp.cn", "soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
    url = url.replace("soul-app.oss-cn-hangzhou.aliyuncs.com","soul-app.oss-cn-hangzhou-internal.aliyuncs.com")
    url = url.replace("bratro.soulapp.cn/app", "soul-audit.oss-cn-hangzhou-internal.aliyuncs.com")
    if ("oss-cn-hangzhou-internal.aliyuncs.com" in url):
        url = url.replace("https", "http")
    return url


resize = 256
resolution = 224

data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(resize),
            transforms.RandomResizedCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop((resolution,resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


max_text_length = 512
tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
model = generate_clip()
model.load_state_dict(torch.load("/mnt/data-nas/cy/llama-study/cy_llama/llama-vit/checkpoints/model_9_342.pth", map_location='cpu')['model_state_dict'])
model = model.cuda()


def infer(text, image_path):

    pattern = r'(\[.*?\]|<innerTag>.*?</innerTag>)'
    text = re.sub(pattern, '', text)
    text_id = tokenizer.encode(text,add_special_tokens=False)[:max_text_length]
    text_id = text_id + (max_text_length-len(text_id)) * [(tokenizer.special_tokens['<eos>'])]
    text_id = torch.LongTensor(text_id).unsqueeze(0)


    response = requests.get(convert_to_internal_url(image_path))
    RGB_image = Image.open(BytesIO(response.content)).convert('RGB')
    image = data_transforms["val"](RGB_image).unsqueeze(0)


    model.eval()

    with torch.no_grad():
        text_emb, image_emb = model(text_id.cuda(), image.cuda())
        
        normalized_tensor_1 = text_emb / text_emb.norm(dim=-1, keepdim=True)
        normalized_tensor_2 = image_emb / image_emb.norm(dim=-1, keepdim=True)
        cos_sim = (normalized_tensor_1 * normalized_tensor_2).sum(dim=-1)

        return cos_sim, RGB_image



demo = gr.Interface(
    fn=infer, 
    inputs=[
        gr.components.Textbox(label="Text Input"),
        gr.components.Textbox(label="Image Input"),
    ],
    outputs=[
        gr.components.Textbox(label="Cos Sim"),
        gr.components.Image(label="Image Display")
    ],
    title="Text-Image"
)


# 启动Gradio应用
if __name__ == "__main__":
    demo.launch(server_name='10.100.15.190', server_port=8882)