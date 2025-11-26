import os
import sys

import re
import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.data import html2str, identify_language
from cs336_data.data import mask_emails, mask_phones, mask_ip
from cs336_data.data import classify_nsfw, classify_toxic_speech, gopher_quality_filter

def parse_args():
    parser = argparse.ArgumentParser(description='benchmark')
    parser.add_argument('--auto', type=int, default=0, help='batch eval model')
    parser.add_argument('--vocab_size', type=int, default=10000, help='vocab_size')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--seq_len', type=int, default=256, help='seq_len')
    parser.add_argument('--d_model', type=int, default=1280, help='d_model')
    parser.add_argument('--d_ff', type=int, default=5120, help='d_ff')
    parser.add_argument('--num_layers', type=int, default=36, help='num_layers')
    parser.add_argument('--num_heads', type=int, default=20, help='num_heads')
    parser.add_argument('--rope_theta', type=int, default=10000, help='rope_theta')
    parser.add_argument('--num_warmups', type=int, default=5, help='num_warmups')
    parser.add_argument('--num_steps', type=int, default=10, help='rope_theta')
    parser.add_argument('--only_forward', type=bool, default=True, help='only_forward')
    parser.add_argument('--use_mixed_precision', type=bool, default=False, help='use_mixed_precision')
    parser.add_argument('--use_memory_profiling', type=bool, default=False, help='memory_profiling')
    return parser.parse_args()

def read_warc_file(warc_path):
    """读取 WARC 文件并提取 HTML 内容"""
    with gzip.open(warc_path, 'rb') as stream:
        for record in ArchiveIterator(stream, record_types=WarcRecordType.response):
            # 获取响应内容（HTML 字节数据）
            html_bytes = record.reader.read()
            
            # 获取 URL
            url = record.headers.get('WARC-Target-URI', 'Unknown URL')
            content_type = record.http_headers.get('Content-Type', '')
            
            # print(f"URL: {url}")
            # print(f"Content Length: {len(html_bytes)} bytes")
            # print(f"content_type: {content_type}")
            # print("-" * 50)
            
            # 返回第一个找到的 HTML 内容（或继续处理所有记录）
            yield html_bytes, url
    
    yield None, None

def clean_extracted_text(text):
    """清理提取的文本，规范化空白字符"""

    # 分割成行处理
    lines = text.splitlines()
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        
        # 过滤掉只有特殊符号的行
        if (re.match(r'^[•\s\.\-]*$', line) or  # 只有•、点、横线、空格
            len(line) <= 1):                    # 只有一个字符的行
            continue
            
        # 处理行首的列表符号
        line = re.sub(r'^\s*[•・‧∙●○▪➢➣➤]\s*', '', line)  # 移除行首的各种列表符号
        
        # 合并连续的列表符号
        line = re.sub(r'([•・‧∙●○▪➢➣➤])\s*([•・‧∙●○▪➢➣➤])+', '•', line)
        
        cleaned_lines.append(line)

    # 重新组合文本
    cleaned_text = '\n'.join(cleaned_lines)
    
    # 进一步规范化空白
    cleaned_text = re.sub(r'\n\s*\n', '\n\n', cleaned_text)  # 合并多个空行
    cleaned_text = re.sub(r'[ \t]+', ' ', cleaned_text)      # 合并空格和制表符
    
    return cleaned_text.strip()

if __name__ == "__main__":
    warc_path = './data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz'
    proc_idx = 0
    proc_num = 100
    for html_bytes, url in read_warc_file(warc_path):
        if html_bytes is not None:
            src_str = html2str(html_bytes)
            src_str = clean_extracted_text(src_str)
            
            # 打印en
            # language, confidence = identify_language(src_str)
            # if language == 'en':
            #     print(proc_idx, src_str[:200].replace('\n', ' ').strip(), language, confidence)
            #     proc_idx += 1

            # 打印email、phone、ip过滤结构
            # rst_str, num_masked_emails = mask_emails(src_str)
            # rst_str, num_masked_phones = mask_phones(rst_str)
            # rst_str, num_masked_ip = mask_ip(rst_str)
            # if num_masked_emails > 0 or num_masked_phones > 0 or num_masked_ip:
            #     print(proc_idx, \
            #           src_str[:200].replace('\n', ' ').strip(), "=>", \
            #           rst_str[:200].replace('\n', ' ').strip())
            #     proc_idx += 1

            # 打印有害内容
            # nsfw_rst, nsfw_conf = classify_nsfw(src_str)
            # toxic_rst, toxic_conf = classify_toxic_speech(src_str)
            # if nsfw_rst == "nsfw" or toxic_rst == "toxic":
            #     print(proc_idx, src_str[:500].replace('\n', ' ').strip(), nsfw_rst, nsfw_conf, toxic_rst, toxic_conf, url)
            #     proc_idx += 1

            # 打印指令过滤
            not_filter_flg = gopher_quality_filter(src_str)
            if not not_filter_flg:
                print(proc_idx, src_str[:200].replace('\n', ' ').strip(), not_filter_flg)
                proc_idx += 1
                    
            if proc_idx >= proc_num:
                break

    
