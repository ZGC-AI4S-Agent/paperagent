import os
import time
from google import genai
from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch

from docagent.qa.base import BaseQA

GEMINI_API_KEY = "AIzaSyB5ZGnyZyWFaLcIAfhVtdqku50xpwnA5FY"

class SingleDocQA(BaseQA):
    def __init__(self):
        super().__init__()
    
    def prefix(self, question, context):
        res = "Question: " + question + "\n"
        if context:
            res += "Document: " + context + "\n"
        res += "Please answer the question above based on the document provided."
        return res
    
    def response(self, request):
        raise NotImplementedError
    
    def answer(self, input_dict):
        question = input_dict['question']
        document = input_dict['local_doc']
        if isinstance(document, str):
            context = document
        request = self.prefix(question, context)
        return self.response(request)


class GeminiSingleDocQA(SingleDocQA):
    def __init__(self):
        super().__init__()
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def response(self, request):
        response = self.client.models.generate_content(model='gemini-2.0-flash-exp', contents=request)
        return response.text
    
    def prefix(self, question, context):
        res = []
        if context:
            if isinstance(context, types.File):
                context = types.Content(
                    role='user',
                    parts=[
                        types.Part.from_uri(
                            file_uri=context.uri,
                            mime_type=context.mime_type
                        )
                    ]
                )
            res.append(context)
            # res.append('\n\n')
        if not question and context:
            question = "请总结以上文件或文本的内容，并用中英双语进行回答。"

        res.append(question)
        return res
    
    def upload_file(self, file_path):
        file = self.client.files.upload(path=file_path)
        while file.state == "PROCESSING":
            # print('Waiting for file to be processed.')
            time.sleep(10)
            file = self.client.files.get(name=file.name)

        if file.state == "FAILED":
            raise ValueError(file.state)
        return file
    
    def answer(self, input_dict):
        question = input_dict['question']
        document = input_dict.get('local_doc', None)
        context = None
        if isinstance(document, str):
            context = document
            if os.path.exists(document):
                context = self.upload_file(document)       
        request = self.prefix(question, context)
        return self.response(request)


class GeminiGoogleSingleDocQA(SingleDocQA):
    def __init__(self):
        super().__init__()
        google_search_tool = Tool(
            google_search = GoogleSearch()
        )
        self.client = genai.Client(api_key=GEMINI_API_KEY)
        self.gen_config = GenerateContentConfig(
            tools=[google_search_tool],
            response_modalities=["TEXT"],
        )

    
    def response(self, request):
        response = self.client.models.generate_content(model='gemini-2.0-flash-exp', contents=request, config=self.gen_config)
        return response.text
    
    def prefix(self, question, context):
        res = []
        if context:
            if isinstance(context, types.File):
                context = types.Content(
                    role='user',
                    parts=[
                        types.Part.from_uri(
                            file_uri=context.uri,
                            mime_type=context.mime_type
                        )
                    ]
                )
            res.append(context)
            # res.append('\n\n')
        if not question and context:
            question = "请总结以上文件或文本的内容，并用中英双语进行回答。"

        res.append(question)
        return res
    
    def upload_file(self, file_path):
        file = self.client.files.upload(path=file_path)
        while file.state == "PROCESSING":
            # print('Waiting for file to be processed.')
            time.sleep(10)
            file = self.client.files.get(name=file.name)

        if file.state == "FAILED":
            raise ValueError(file.state)
        return file
    
    def answer(self, input_dict):
        question = input_dict['question']
        document = input_dict.get('local_doc', None)
        context = None
        if isinstance(document, str):
            context = document
            if os.path.exists(document):
                context = self.upload_file(document)       
        request = self.prefix(question, context)
        return self.response(request)
