import os
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

import re
import fasttext
from xopen import xopen
from collections import defaultdict
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding

import nltk
# nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

import mmh3
import unicodedata

# 模型初始化
model_path = "./data/lid.176.bin" 
lang_model = fasttext.load_model(model_path)
model_path = "./data/jigsaw_fasttext_bigrams_nsfw_final.bin" 
nsfw_model = fasttext.load_model(model_path)
model_path = "./data/jigsaw_fasttext_bigrams_hatespeech_final.bin" 
toxic_speech_model = fasttext.load_model(model_path)
model_path = "./data/fasttext_quality_v1.bin" 
quality_model = fasttext.load_model(model_path)

def html2str(
    html_bytes: bytes
) -> str | None:
    # 步骤 1：检测编码
    encoding = detect_encoding(html_bytes)
    # 添加 errors 参数处理解码错误
    html_str = html_bytes.decode(encoding, errors='ignore')
    
    # 步骤 2：按正确编码提取文本
    clean_text = extract_plain_text(html_str, encoding)

    # # 尝试不同的参数组合
    # clean_text = extract_plain_text(
    #     html_str, 
    #     encoding,
    #     main_content=True,      # 只提取主要内容
    #     # ignore_images=True,      # 忽略图片相关文本
    #     # links=False,             # 不保留链接文本
    #     # preserve_formatting=False # 不保留格式，更紧凑
    # )
    return clean_text

def fasttext_classify(text: str, model) -> tuple[Any, float]:
    text = text.replace('\n', ' ').strip()
    predictions = model.predict(text, k=1)  # k=1 返回最可能的一种语言
    language = predictions[0][0].replace('__label__', '')
    confidence = predictions[1][0]
    # print(language, confidence)
    return language, confidence

def identify_language(text: str) -> tuple[Any, float]:
    return fasttext_classify(text, lang_model)

def classify_nsfw(text: str) -> tuple[Any, float]:
    return fasttext_classify(text, nsfw_model)

def classify_toxic_speech(text: str) -> tuple[Any, float]:
    return fasttext_classify(text, toxic_speech_model)

def classify_quality(text: str) -> tuple[Any, float]:
    return fasttext_classify(text, quality_model)
    
def mask_emails(text: str) -> tuple[str, int]:
    # 有锚点^$时，只有整个字符串完全符合模式的才会匹配
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+'
    # email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    print(re.findall(email_pattern, "me at test@gmail.com if"))
    num_masked = len(re.findall(email_pattern, text))
    text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)
    print(num_masked, text)
    return text, num_masked

def mask_emails(text: str) -> tuple[str, int]:
    # 有锚点^$时，只有整个字符串完全符合模式的才会匹配
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+'
    # email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    # print(re.findall(email_pattern, "me at test@gmail.com if"))
    num_masked = len(re.findall(email_pattern, text))
    text = re.sub(email_pattern, '|||EMAIL_ADDRESS|||', text)
    # print(num_masked, text)
    return text, num_masked

def mask_phones(text: str) -> tuple[str, int]:
    # phones_pattern = r'[0-9()]+[0-9() -]*[0-9]+'
    
    # 国际电话号码正则表达式
    # phones_pattern = r'''
    #     (?:\+?(\d{1,3}))?            # 国家代码（可选+开头，1-3位数字）
    #     [\s\-\.]?                    # 可选分隔符（空格、横线、点）
    #     (\(?\d{1,4}\)?)?            # 地区代码（可选括号，1-4位数字）
    #     [\s\-\.]?                    # 可选分隔符
    #     (\d{2,4})?                   # 第一部分本地号码（2-4位）
    #     [\s\-\.]?                    # 可选分隔符
    #     (\d{2,4})?                   # 第二部分本地号码（2-4位）
    #     [\s\-\.]?                    # 可选分隔符
    #     (\d{2,4})                    # 第三部分本地号码（2-4位）
    # '''
    phones_pattern = r'(\+\d{1,3}[\- ]?)?\(?\d{3,}\)?[\- ]?[0-9]{3,}[\- ]?[0-9]{3,}'
    # phones_pattern = r'\+?(\d{1,3})[\s\-\.]?(\d{1,5})[\s\-\.]?(\d{4,8})'
    num_masked = len(re.findall(phones_pattern, text))
    text = re.sub(phones_pattern, '|||PHONE_NUMBER|||', text)
    # print(num_masked, text)
    return text, num_masked

def mask_ip(text: str) -> tuple[str, int]:
    ip_pattern = r'[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+'
    num_masked = len(re.findall(ip_pattern, text))
    text = re.sub(ip_pattern, '|||IP_ADDRESS|||', text)
    # print(num_masked, text)
    return text, num_masked

def gopher_quality_filter(text: str) -> bool:
    # 分割成行处理
    lines = text.splitlines()

    # word
    word_num = 0
    word_length = 0
    blank_num = 0
    line_total_num = 0
    line_ellipsis_num = 0
    for line in lines:
        line_total_num += 1
        if line.endswith("..."):
            line_ellipsis_num += 1
        tokens = word_tokenize(line)
        word_num += len(tokens)
        blank_num += sum(1 for token in tokens if not token or not token.strip())
        word_length += sum(len(token) for token in tokens)

    # print(word_num, word_length, line_ellipsis_num, line_total_num, blank_num, word_num)
    if word_num < 50 or word_num > 100000:
        return False
    if word_length / word_num < 3 or word_length / word_num > 10:
        return False
    if line_ellipsis_num / line_total_num > 0.3:
        return False
    if blank_num / word_num > 0.2:
        return False
    return True
        
def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    hash_dict = defaultdict(int)
    for input_file in input_files:
        with open(input_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                hash_dict[hash(line)] += 1
                
    for input_file in input_files:
        dst_path = os.path.join(output_directory, os.path.basename(input_file))
        dst_lines = list()
        with open(input_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                if hash_dict[hash(line)] == 1:
                    dst_lines.append(line)
        with open(dst_path, 'w') as fp:
            for line in dst_lines:
                fp.write(line)
            

def normalize_text(text: str) -> str:
    """规范化文本：小写、去标点、规范化空格、去重音、Unicode 规范化"""
    # 转换为小写
    text = text.lower()
    
    # 移除标点符号（保留单词间的空格）
    text = re.sub(r'[^\w\s]', '', text)
    
    # 规范化空格（多个空格合并为一个）
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Unicode NFD 规范化并去除重音
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    return text

def generate_ngrams(text: str, n: int):
    """生成文本的 n-gram 集合"""
    words = word_tokenize(text)
    ngrams = set()
    if len(words) < n:
        ngrams.add(' '.join(words))
        return ngrams
    
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i + n])
        ngrams.add(ngram)
    
    return ngrams

def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    def compute_jaccard(A, B):
        intersection = len(A & B)  # @inspect intersection
        union = len(A | B)  # @inspect union
        return intersection / union

    rows_per_band = num_hashes // num_bands
    buckets = defaultdict(set)
    remove_dict = defaultdict(int)
    ngrams_dict = defaultdict(set)
    for input_file in input_files:
        with open(input_file, 'r') as fp:
            content = fp.read()
            text = normalize_text(content)
            ngrams_list = generate_ngrams(text, ngrams)
            ngrams_dict[input_file] = ngrams_list
            
            signature = []
            for idx in range(num_hashes):
                signature.append(min(mmh3.hash(cur_ngrams, idx) for cur_ngrams in ngrams_list))
                
            for band_idx in range(num_bands):
                start = band_idx * rows_per_band
                end = start + rows_per_band
                band_signature = tuple(signature[start:end])
                
                # 创建桶键（条带索引 + 条带签名哈希）
                bucket_key = (band_idx, band_signature)
                buckets[bucket_key].add(input_file)

    # 查找候选重复对
    for buckets_files in buckets.values():
        if len(input_files) > 1:
            buckets_files = list(buckets_files)
            for i in range(len(buckets_files)):
                for j in range(i + 1, len(buckets_files)):
                    similarity = compute_jaccard(ngrams_dict[buckets_files[i]], ngrams_dict[buckets_files[j]])
                    # print(similarity, buckets_files[i], buckets_files[j])
                    if similarity >= jaccard_threshold:
                        # 保留第一个、删除第二个
                        # remove_dict[buckets_files[i]] += 1
                        remove_dict[buckets_files[j]] += 1              

    # 写入
    # print(remove_dict)
    for input_file in input_files:
        # print(input_file)
        if input_file in remove_dict.keys():
            continue
        dst_path = os.path.join(output_directory, os.path.basename(input_file))
        dst_lines = list()
        with open(input_file, 'r') as fp:
            lines = fp.readlines()
            for line in lines:
                dst_lines.append(line)
                    
        # print(input_file, dst_path)            
        with open(dst_path, 'w') as fp:
            for line in dst_lines:
                fp.write(line)
                    


                
                
                

    
        

    