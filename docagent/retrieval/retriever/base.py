class BaseRetriever:
    def __init__(self, embedding, database):
        self.embedding = embedding
        self.database = database
    
    def add(self, data):
        raise NotImplementedError

    def retrieve(self, query: str):
        raise NotImplementedError