import os
os.environ['CUDA_VISIBLE_DEVICES'] = "7"

import gradio as gr
import torch
from model import ModelArgs, Transformer
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer


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

    model.load_state_dict(torch.load("/mnt/data-nas/cy/llama-study/cy_llama/llama-vit/checkpoints/model_sft_9.pth", map_location="cpu")['model_state_dict'])
    tokenizer=ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    return model, tokenizer


model, tokenizer = init_model()
model.cuda()


def generate_text(prompt):
    data = {"question": prompt}
    prompt=data['question']
    x=tokenizer.encode(prompt,add_special_tokens=False)
    x = (torch.tensor(x, dtype=torch.long, device='cuda')[None, ...])
    with torch.no_grad():
        y = model.generate(x, 2, max_new_tokens=100, temperature=1.0, top_k=100)
        answer=tokenizer.decode(y[0].tolist())
        answer=answer.replace(prompt,'')

    return answer


demo = gr.Interface(
    fn=generate_text, 
    inputs="text", 
    outputs="text", 
    title="CY-GPT",
    description="文本语言模型"
)

# 启动Gradio应用
if __name__ == "__main__":
    demo.launch(server_name='10.100.15.190', server_port=8881)