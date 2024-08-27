import os
import requests
import io
import csv
from tqdm import tqdm, trange
import random
import multiprocessing as mp
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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

all_data = []
reader = csv.reader(open("/mnt/data-nas/cy/llama-study/cy_llama/llama-vit/data/csv/coca_query_1195694.csv"))
next(reader)
for item in tqdm(reader):
    all_data.append(item[2])

imgs = list(all_data)
random.shuffle(imgs)
print("图片数量:{}".format(len(imgs)))
# exit()
num_procs = 96
num_ = len(imgs) // num_procs 
save_path = "/mnt/data-nas/cy/llama-study/cy_llama/llama-vit/data/images"
def run(start, end):
    global imgs
    for i in trange(start, end):
        
        # 直接下载逻辑
        try:
            url = imgs[i]
            img_bin = requests.get(url).content
            bytes_obj = io.BytesIO(img_bin)
            image = Image.open(bytes_obj)
            img = image.convert('RGB')
            w, h = img.size
            radio = w/h
            if radio<1:
                ret = img.resize((256, int(h/w*256)))
            else:
                ret = img.resize((int(w/h*256), 256))
            os.makedirs(save_path, exist_ok=True)
            ret.save(save_path + "/" + os.path.basename(url))
        except:
            try:
                url = convert_to_internal_url(imgs[i])
                if url.endswith(".gif"): #gif取第一张      
                    res = requests.get(url,timeout=5)
                    img_bin = res.content
                    im = Image.open(io.BytesIO(img_bin))
                    im.seek(0)
                    imgByteArr = io.BytesIO()
                    im.save(imgByteArr,"PNG")
                    img_bin =  imgByteArr.getvalue()
                else:
                    #img = Image.open(imgurl).convert('RGB')
                    r = requests.get(url,stream=True)
                    tmp_d = next(r.iter_content(chunk_size=15))
                    if b'\x00\x00\x00' in tmp_d[:10]:
                        url += '?x-oss-process=image/resize,m_fill,h_5000,w_5000/format,png/quality,q_80'
                    try:
                        if str(tmp_d[:4],encoding='utf-8') == "RIFF":
                            res = requests.get(url).content
                            im = Image.open(io.BytesIO(res))
                            imgByteArr = io.BytesIO()
                            im.save(imgByteArr,"PNG")
                            img_bin = imgByteArr.getvalue()
                        elif str(tmp_d[:4],encoding='utf-8') != "RIFF":
                            img_bin = requests.get(url).content
                    except:
                        pass
                    img_bin = requests.get(url).content
                bytes_obj = io.BytesIO(img_bin)
                image = Image.open(bytes_obj)
                img = image.convert('RGB')
                w, h = img.size
                radio = w/h
                if radio<1:
                    ret = img.resize((256, int(h/w*256)))
                else:
                    ret = img.resize((int(w/h*256), 256))
                os.makedirs(save_path, exist_ok=True)
                ret.save(save_path + "/" + os.path.basename(url))
            except Exception as e:
                print(e)


process = []
for i in range(num_procs):
    start = i* num_
    end = start+ num_
    if i == num_procs -1 :end = len(imgs)
    p = mp.Process(target=run,args=(start,end))
    p.start()
    process.append(p)
for p in process:p.join()