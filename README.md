<h2 align="center">Ocean-OCR</a></h2>

<p align="center">
<img src="benchmarks.png" style="width: 700px" align=center>
</p>


## Introduction
Multimodal large language models (MLLMs) have shown impressive capabilities across various domains, excelling in processing and understanding information from multiple modalities. Despite the rapid progress made previously, insufficient OCR ability hinders MLLMs from excelling in text-related tasks. In this paper, we present Ocean-OCR, a 3B MLLM with state-of-the-art performance on various OCR scenarios and comparable understanding ability on general tasks. We employ Native Resolution ViT to enable variable resolution input and utilize a substantial collection of high-quality OCR datasets to enhance the model performance.

## OCR practical scenarios

(1) Document Extraction
```
python eval_internvl2_5.py --checkpoint_path your_checkpoint_path --eval_type document_en
```
```
python eval_internvl2_5.py --checkpoint_path your_checkpoint_path --eval_type document_zh
```
(2) Scene Text Recognition
```
python eval_internvl2_5.py --checkpoint_path your_checkpoint_path --eval_type scene_text_rec
```
(3) Handwritten Recognition
```
python eval_internvl2_5.py --checkpoint_path your_checkpoint_path --eval_type handwritten_en
```
```
python eval_internvl2_5.py --checkpoint_path your_checkpoint_path --eval_type handwritten_zh
```
