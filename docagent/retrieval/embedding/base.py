class BaseEmbedding:
    def __init__(self):
        # set up connnection or load checkpoints
        pass
    def parse(self, x):
        # parse the input data into text
        return x
    
    def embedding_text(self, x):
        # embedding the input text
        raise NotImplementedError

    def postprocess(self, x):
        return x

    def embedding(self, x):
        # do parsing
        parsed = self.parse(x)
        # do embedding
        embed = self.embedding_text(parsed)
        # postprocess
        post = self.postprocess(embed)
        return post



        