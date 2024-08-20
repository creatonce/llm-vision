import json
import glob
import csv
import numpy as np
from tqdm import tqdm
from chatglm_tokenizer.tokenization_chatglm import ChatGLMTokenizer

def process_soul():
    reader = csv.reader(open("/mnt/data-nas/cy/code/llama-vit/Baichuan-7B/data/train_val/data.csv"))
    doc_ids=[]
    for item in tqdm(reader):
        pid, text = item
        text_id = tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/soul.bin','wb') as f:
        f.write(arr.tobytes())

def process_wiki_clean():
    with open('./data/wikipedia-cn-20230720-filtered.json','r',encoding='utf-8') as f:
        data=json.load(f)
    doc_ids=[]
    for line in tqdm(data):
        text=line['completion']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./data/wiki.bin','wb') as f:
        f.write(arr.tobytes())


if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='./chatglm_tokenizer/tokenizer.model')
    # 数据预处理-如果下载分词处理后的数据，可以不用执行以下函数
    process_wiki_clean()
    # process_soul()

    print('data processing finished!')
    # exit()

    # 分词处理后的文件列表
    data_path_list=[
        # './data/soul.bin',
        './data/wiki.bin'
    ]
    data_lst=[]
    for data_path in tqdm(data_path_list):
        with open(data_path,'rb') as f:
            data=np.fromfile(f,dtype=np.uint16)
            data_lst.append(data)
    arr = np.concatenate(data_lst)
    print(arr.shape)
    with open('./data/data.bin','wb') as f:
        f.write(arr.tobytes())
