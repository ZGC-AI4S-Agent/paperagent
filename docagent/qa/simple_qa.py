from google import genai

from docagent.qa.base import BaseQA

GEMINI_API_KEY = "AIzaSyA7mrCp8x0kd6C-WBP6-PL6HgbaaPEYJls"

class SimpleQA(BaseQA):
    def __init__(self):
        super().__init__()
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def response(self, request):
        response = self.client.models.generate_content(model='gemini-2.0-flash-exp', contents=request)
        return response.text
    
    def answer(self, input_dict):
        request = input_dict['question']
        return self.response(request)