import re

from docagent.parser.base import BaseParser
from docagent.parser.utils import download_url


class SimpleStrParser(BaseParser):
    def __init__(self):
        super().__init__()
    
    def parse(self, x):
        # find url
        urls = re.findall(r'(https?://[^\s]+)', x)
        # find question
        question = x
        doc = None
        if len(urls) > 0:
            # download the file
            fp = download_url(urls[0], 'tmp')
            doc = fp
            question = question.replace(urls[0], '')
        # setup request
        res = dict()
        res['question'] = question
        if doc:
            res['local_doc'] = doc
        return res