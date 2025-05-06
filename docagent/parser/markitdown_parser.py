import os
import re
from markitdown import MarkItDown

from docagent.parser.base import BaseParser
# from qa.simple_qa import SimpleQA
from docagent.parser.utils import download_url

class MarkItDownParser(BaseParser):
    def __init__(self):
        super().__init__()
        self.md = MarkItDown()
        # self.qa = SimpleQA()

    def parse(self, fp):
        if not os.path.exists(fp):
            # check if the fp is url
            # maybe use llm to extract the question and context
            urls = re.findall(r'(https?://[^\s]+)', fp)
            if len(urls) > 0:
                # download the file
                fp = download_url(urls[0], 'tmp')
            else:
                return fp
            
        result = self.md.convert(fp)
        return result.text_content