import os
import sys

import re
import gzip
from pathlib import Path
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.data import html2str, identify_language
from cs336_data.data import mask_emails, mask_phones, mask_ip
from cs336_data.data import classify_nsfw, classify_toxic_speech, gopher_quality_filter, classify_quality
from cs336_data.data import exact_line_deduplication, minhash_deduplication
from cs336_data.benchmark import read_warc_file, clean_extracted_text

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

if __name__ == "__main__":
    warc_path = './data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    dst_dedup_pre_dir = './data/dedup_pre_ch'
    dst_exact_line_dedup_dir = './data/exact_line_dedup_ch'
    dst_minhash_dedup_dir = './data/minhash_dedup_ch'
    output_path = './data/minhash_dedup_ch.np'
    for dst_dir in [dst_dedup_pre_dir, dst_exact_line_dedup_dir, dst_minhash_dedup_dir]:
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

    proc_idx = 0
    warc_idx = 0
    filter_flg = True
    dedup_flg = True
    tokenize_flg = True
    
    if filter_flg:
        for html_bytes, url in read_warc_file(warc_path):
            warc_idx += 1
            if html_bytes is not None:
                src_str = html2str(html_bytes)
                src_str = clean_extracted_text(src_str)
                language, confidence = identify_language(src_str)
                # print(src_str[:200].replace('\n', ' ').strip(), language, confidence)
                if language != 'zh':
                    continue
    
                # 有害内容
                nsfw_rst, nsfw_conf = classify_nsfw(src_str)
                toxic_rst, toxic_conf = classify_toxic_speech(src_str)
                # if nsfw_rst == "nsfw" or toxic_rst == "toxic":
                if nsfw_rst != "nsfw" and toxic_rst != "toxic":
                    continue
    
                # # 质量过滤
                not_filter_flg = gopher_quality_filter(src_str) 
                if not not_filter_flg:
                    continue
                not_filter_flg = classify_quality(src_str) 
                if not not_filter_flg:
                    continue
    
                # # email、phone、ip
                rst_str, num_masked_emails = mask_emails(src_str)
                rst_str, num_masked_phones = mask_phones(rst_str)
                rst_str, num_masked_ip = mask_ip(rst_str)
    
                # 写入
                dst_txt_path = os.path.join(dst_dedup_pre_dir, str(warc_idx) + '.txt')
                print(proc_idx, dst_txt_path)
                with open(dst_txt_path, 'w') as fp:
                    fp.write(rst_str)
                proc_idx += 1

    if dedup_flg:
        exact_line_deduplication(
            list(Path(dst_dedup_pre_dir).glob("*.txt")), 
            dst_exact_line_dedup_dir
        )
        minhash_deduplication(
            input_files=list(Path(dst_exact_line_dedup_dir).glob("*.txt")),
            output_directory=dst_minhash_dedup_dir,
            num_hashes=100,
            num_bands=10,
            ngrams=5,
            jaccard_threshold=0.8
        )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    def tokenize_line_and_add_eos(line):
        return tokenizer.encode(line) + [tokenizer.eos_token_id]
    
    if tokenize_flg:
        results = list()
        all_files = list(Path(dst_minhash_dedup_dir).glob("*.txt"))
        for filepath in tqdm(all_files):
            with open(filepath, 'r') as fp:
                lines = fp.readlines()
            for line in lines:
                result = tokenize_line_and_add_eos(line)
                results.append(result)
            # break
        all_idx = [token_id for result in results for token_id in result]
        idx_array = np.array(all_idx, dtype=np.uint16)
        idx_array.tofile(output_path)
        # idx_array: (13311064,)
        print("idx_array:", idx_array.shape)
                
                
            
            
        
        
            

    
