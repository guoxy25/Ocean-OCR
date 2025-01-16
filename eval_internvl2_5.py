from transformers import AutoModelForCausalLM, AutoTokenizer
from argparse import ArgumentParser
import torch
torch.cuda.set_device(1)
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


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, 
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--eval-type", type=str, default='document_zh')
    args = parser.parse_args()
    return args

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values



def load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, trust_remote_code=True, use_fast=False)

    device_map = 'cuda'
    model = AutoModel.from_pretrained(
        args.checkpoint_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True).eval().cuda()

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

    def extract_number(file_name):
        match = re.search(r'(\d+)\.png$', file_name)
        if match:
            return int(match.group(1))
        return 0  # 如果没有匹配到数字，返回0
    
    text_loss_list, img_loss_list = [],[]
    with open(gt_path, 'r') as f:
        gt_list = json.load(f)

    out_list = []
    with torch.inference_mode():
        for data in tqdm.tqdm(gt_list):
            out_dict= {}
            out_dict['image'] = data['image']
            out_dict['label'] = data['conversations'][1]['value']
            imgname = os.path.join(img_dir, data['image'])
            pixel_values = load_image(imgname, max_num=12).to(torch.bfloat16).cuda()
            generation_config = dict(max_new_tokens=1024, do_sample=True)
            question = '<image>\nPlease output the text in the image directly.'
            response = model.chat(tokenizer, pixel_values, question, generation_config)
            out_dict['answer'] = response
            print(f'User: {question}\nAssistant: {response}')
            out_list.append(out_dict)
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(out_list, file, ensure_ascii=False, indent=4)
    doc_text_eval(args.eval_type, out_list)

    
            

