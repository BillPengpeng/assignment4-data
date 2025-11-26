import os
import sys

import re
import gzip
import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.data import html2str, identify_language
from cs336_data.data import mask_emails, mask_phones, mask_ip
from cs336_data.data import classify_nsfw, classify_toxic_speech, gopher_quality_filter
from cs336_data.benchmark import read_warc_file, clean_extracted_text

def proc_gz(rst_lines, gz_path, gz_label, gz_sample_num, sample_neg=True):
    proc_idx = 0
    for html_bytes, url in read_warc_file(gz_path):
        if html_bytes is not None:
            src_str = html2str(html_bytes)
            src_str = clean_extracted_text(src_str)
            language, confidence = identify_language(src_str)
            if language != 'en':# or confidence < 0.7:
                continue
            
            # nsfw_rst, nsfw_conf = classify_nsfw(src_str)
            # toxic_rst, toxic_conf = classify_toxic_speech(src_str)
            not_filter_flg = gopher_quality_filter(src_str)
            
            # if (nsfw_rst == "nsfw" and nsfw_conf > 0.99) or \
            #    (toxic_rst == "toxic" and toxic_conf > 0.99) or \
            #    not not_filter_flg:
            if sample_neg and (not not_filter_flg) and len(src_str.strip()) > 0:
                rst_lines.append(gz_label + ' ' + src_str.strip())
                print(proc_idx, src_str[:500].replace('\n', ' ').strip(), len(src_str.strip()))
                proc_idx += 1

            if (not sample_neg): # and not_filter_flg and len(src_str.strip()) > 0:
                rst_lines.append(gz_label + ' ' + src_str.strip())
                print(proc_idx, src_str[:500].replace('\n', ' ').strip(), len(src_str.strip()))
                proc_idx += 1

            if proc_idx >= gz_sample_num:
                break

if __name__ == "__main__":
    rst_lines = list()
    gz_path = './data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    gz_label = '__label__cc'
    gz_sample_num = 1000
    print(f"start proc {gz_path}...")
    proc_gz(rst_lines, gz_path, gz_label, gz_sample_num, sample_neg=True)
    print(f"proc {gz_path} over!")

    gz_path = './data/enwiki-20240420-extracted_urls_sample_v1.warc.warc.gz'
    gz_label = '__label__wiki'
    gz_sample_num = 1000
    print(f"start proc {gz_path}...")
    proc_gz(rst_lines, gz_path, gz_label, gz_sample_num, sample_neg=False)
    print(f"proc {gz_path} over!")

    dst_fasttext_dataset = './data/fasttext_trainset_v1.txt'
    with open(dst_fasttext_dataset, 'w') as fp:
        for line in rst_lines:
            fp.write(line + '\n')
    print(f"write {dst_fasttext_dataset} over!")

    # 训练模型
    model = fasttext.train_supervised(
        input=dst_fasttext_dataset,
        label='__label__',  # 标签前缀
        dim=100,            # 词向量维度
        epoch=25,           # 训练轮数
        lr=0.1,             # 学习率
        wordNgrams=2,       # 使用2-gram特征
        loss='softmax',     # 损失函数
        thread=4,           # 线程数
        verbose=2          # 打印训练信息
    )
    
    # 保存模型
    model_save_path = "./data/fasttext_quality_v1.bin"
    model.save_model(model_save_path)
    print(f"模型已保存至: {model_save_path}")

    # 在训练集上评估模型
    result = model.test(dst_fasttext_dataset)
    print(f"训练集评估 - 样本数: {result[0]}, 精确率: {result[1]:.4f}, 召回率: {result[2]:.4f}")
            
