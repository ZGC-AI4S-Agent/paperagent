import os

import requests
from urllib.parse import urlparse
import hashlib

def download_url(url, download_dir):
    os.makedirs(download_dir, exist_ok=True)

    response = requests.get(url, stream=True)

    file_name = os.path.basename(urlparse(url).path)
    file_ext = file_name.split('.')[-1].lower() if '.' in file_name else None
    
    downloadable_extensions = ['pdf', 'doc', 'docx', 'ppt', 'pptx', 'xls', 'xlsx', 'txt']

    md = hashlib.md5()
    md.update(url.encode('utf-8')) 
    md.hexdigest()

    if file_ext in downloadable_extensions:
        filename = f"{md.hexdigest()}.{file_ext}"
        fp = os.path.join(download_dir, filename)
        with open(fp, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
    else:
        filename = f"{md.hexdigest()}.txt"
        fp = os.path.join(download_dir, filename)
        jina_url = f"https://r.jina.ai/{url}"
        jina_response = requests.get(jina_url)
        with open(fp, 'w', encoding='utf-8') as file:
            file.write(jina_response.text) 
    return fp