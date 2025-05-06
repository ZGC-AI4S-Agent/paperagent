class BaseQA:
    def __init__(self, ):
        pass

    def prefix(self, question, context):
        return question
    
    def answer(self, input_dict):
        raise NotImplementedError