import os
import time
import json
from google import genai
from google.genai import types

from docagent.qa.base import BaseQA

GEMINI_API_KEY = "AIzaSyA7mrCp8x0kd6C-WBP6-PL6HgbaaPEYJls"

class MultipleDocQA(BaseQA):
    def __init__(self):
        super().__init__()
    
    def prefix(self, question, document, context_docs):
        res = "Question: " + question + "\n\n"
        if document:
            res += "Document: " + document + "\n\n"
        
        if context_docs:
            res += "Related Documents: " + str(context_docs) + "\n"

        res += "Please answer the question above based on the document provided."
        return res
    
    def response(self, request):
        raise NotImplementedError
    
    def answer(self, input_dict):
        question = input_dict['question']
        document = input_dict['local_doc']
        context_docs = input_dict['local_doc']
        if isinstance(document, str):
            context = document
        request = self.prefix(question, context, context_docs)
        return self.response(request)

class RetrievalMultipleDocQA(MultipleDocQA):
    def __init__(self, retriever):
        super().__init__()
        self.retriever = retriever
    
    def response(self, request):
        raise NotImplementedError
    
    def retrieval(self, query):
        raise NotImplementedError
    
    def answer(self, input_dict):
        question = input_dict['question']
        document = input_dict['local_doc']
        context_docs = self.retrieval(question) + self.retrieval(document)
        if isinstance(document, str):
            context = document
        request = self.prefix(question, context, context_docs)
        return self.response(request)
    

class GeminiRetrievalMultipleDocQA(RetrievalMultipleDocQA):
    def __init__(self, retriever):
        super().__init__(retriever)
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def response(self, request):
        response = self.client.models.generate_content(model='gemini-2.0-flash-exp', contents=request)
        return response.text
    
    def prefix(self, question, context):
        question_retrevial = self.retrieval(question)
        doc_retrevial = []
        if context:
            doc_sum = self.summulize(context)
            doc_retrevial = self.retrieval(doc_sum)
        id_cache = set()
        related = []
        for doc in doc_retrevial:
            if doc['id'] not in id_cache:
                data = dict()
                data['title'] = doc['title']
                data['abstract'] = doc['abstract']
                data['authors'] = doc['authors']
                related.append(data)
                id_cache.add(doc['id'])
        for doc in question_retrevial:
            if doc['id'] not in id_cache:
                data = dict()
                data['title'] = doc['title']
                data['abstract'] = doc['abstract']
                data['authors'] = doc['authors']
                related.append(data)
                id_cache.add(doc['id'])
        res = ['Your goal is to provide a well-reasoned and concise answer to the given Question by primarily relying on the Main Content, while integrating and cross-referencing relevant information from the Related Content. Ensure that your response is clear, logically structured, and accurately reflects the evidence provided.']
        if context:
            res.append("Main content:")
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
        if related:
            res.append("Related content:")
            res.append(json.dumps(related))
            # res.append('\n\n')
        if not question:
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
    
    def summulize(self, document):
        res = []
        if isinstance(document, types.File):
            context = types.Content(
                role='user',
                parts=[
                    types.Part.from_uri(
                        file_uri=document.uri,
                        mime_type=document.mime_type
                    )
                ]
            )
            res.append(context)
        question = "Please summarize the content above within 300 words."
        res.append(question)
        return self.response(res)

    
    def retrieval(self, query):
        res = self.retriever.retrieve(query, top_k=3)
        print(res)
        return res
        
    
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
