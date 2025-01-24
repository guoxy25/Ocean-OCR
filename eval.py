from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
import torch
torch.cuda.set_device(0)
import json
import tqdm
import glob
import random
import os
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode


import json
import nltk
from nltk.metrics import precision, recall, f_measure
import jieba
from nltk.translate import meteor_score


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--eval-type", type=str, default='handwritten_en')
    args = parser.parse_args()
    return args


def load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )
    device_map = 'cuda'

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
        torch_dtype=torch.bfloat16
    )

    return model, tokenizer


def contain_chinese_string(text):
    chinese_pattern = re.compile(r'[\u4e00-\u9fa5]')
    return bool(chinese_pattern.search(text))

def cal_per_metrics(pred, gt):

    metrics = {}

    if contain_chinese_string(gt) or contain_chinese_string(pred):
        reference = jieba.lcut(gt)
        hypothesis = jieba.lcut(pred)
    else:
        reference = gt.split()
        hypothesis = pred.split()

    metrics["bleu"] = nltk.translate.bleu([reference], hypothesis)
    metrics["meteor"] = meteor_score.meteor_score([reference], hypothesis)

    reference = set(reference)
    hypothesis = set(hypothesis)
    metrics["f_measure"] = f_measure(reference, hypothesis)

    metrics["precision"] = precision(reference, hypothesis)
    metrics["recall"] = recall(reference, hypothesis)
    metrics["edit_dist"] = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return metrics

def lable_norm(eval_type, pred):
    if 'scene' in eval_type:
        # 去除一些额外的符号
        pred = pred.replace('\\text{', '').replace('}', '').replace('\n', ' ').replace('-', ' ').replace('*', ' ')
        pred = pred.strip()
        # 根据一些符号划分开并排序
        pred = re.split(r'[（）()，,\s]+', pred)
        pred.sort()
        pred = [p for p in pred if (p != ' ') and (p != '')]
        pred = ' '.join(pred)
    if 'handwritten' in eval_type:
        pred = pred.replace('*', ' ').replace('\n', ' ')
    return pred

def doc_text_eval(eval_type, predicts):

    result = []
    for ann in predicts:
        ann['label'] = lable_norm(eval_type, ann['label'])
        ann['answer'] = lable_norm(eval_type, ann['answer'])
        ans = cal_per_metrics(ann["label"], ann["answer"])
        result.append(ans)

    mean_dict = {}

    mean_dict["eval question num"] = len(result)
    for k, v in result[0].items():
        mean_dict[k] = 0
    
    for each in result:
        for k, v in each.items():
            if v == None:
                v = 0
            mean_dict[k] += v

    for k, v in mean_dict.items():
        if k == "eval question num":
            continue
        mean_dict[k] /= len(result)
    print(json.dumps(mean_dict, indent=4))

if __name__ == "__main__":
    args = _get_args()
    gt_path = 'OCR_eval_data/' + args.eval_type + '.json'
    img_dir = os.path.join('OCR_eval_data', args.eval_type)
    output_file = args.checkpoint_path.split('/')[-1] + '_' + args.eval_type + '.json'

    model, tokenizer = load_model_tokenizer(args)
    processor = model.bind_processor(tokenizer, training=False)

    
    with open(gt_path, 'r') as f:
        gt_list = json.load(f)

    out_list = []
    with torch.inference_mode():
        for data in tqdm.tqdm(gt_list):
            out_dict= {}
            out_dict['image'] = data['image']
            out_dict['label'] = data['conversations'][1]['value']
            imgname = os.path.join(img_dir, data['image'])
            text = f"<img_start_baichuan>{{\"local\": \"{imgname}\"}}<img_end_baichuan>"
            prompt = 'Can you pull all textual information from the images?'
            ret = model.processor('<C_Q>'+text+prompt+'<C_A>')
            input_ids = ret.input_ids
            images = ret.images
            ret = model.generate(
                inputs=torch.LongTensor([input_ids]).cuda(),
                images=[torch.tensor(i, dtype=torch.float32).cuda() for i in images],
                attention_mask=None,
                labels=None,
                audios=None,
                encoder_length=ret.encoder_length.cuda() if ret.encoder_length is not None else None,
                bridge_length=ret.bridge_length.cuda() if ret.bridge_length is not None else None,
                max_new_tokens=2048, do_sample=False, top_k=5, top_p=0.85, temperature=0,
                num_return_sequences=1, repetition_penalty=1.05,
                use_cache=False,
                images_grid=ret.images_grid
            )
            outputs = ''
            for res in ret:
                output = tokenizer.decode(res, skip_special_tokens=True)
                outputs += output

            out_dict['answer'] = outputs
            print(f'User: {prompt}\nAssistant: {output}')
            out_list.append(out_dict)
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(out_list, file, ensure_ascii=False, indent=4)
    doc_text_eval(args.eval_type, out_list)

    
            

