import os
import sys

import re
import gzip
from fastwarc.warc import ArchiveIterator, WarcRecordType
from cs336_data.data import html2str, identify_language
from cs336_data.data import mask_emails, mask_phones, mask_ip
from cs336_data.data import classify_nsfw, classify_toxic_speech, gopher_quality_filter


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

if __name__ == "__main__":
    src_path = './data/enwiki-20240420-extracted_urls.txt.gz'
    # v1
    sample_idx = 0
    sample_num = 1000
    dst_path = './data/enwiki-20240420-extracted_urls_sample_v1.txt'
    rst_lines = list()
    with gzip.open(src_path, 'rb') as stream:
        lines = stream.readlines()
        # print("lines:", len(lines))
        size = len(lines)
        for idx in range(0, size, 100):
            rst_lines.append(lines[idx])
            sample_idx += 1
            if sample_idx >= sample_num:
                break

    with open(dst_path, 'wb') as fp:
        for line in rst_lines:
            fp.write(line)
            
        
    
