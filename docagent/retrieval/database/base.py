class BaseDatabase:
    def __init__(self, name, dimension):
        self.name = name
        self.dimension = dimension
        # 类的定义：接收两个参数name（数据库名称）和dimension（数据的维度）


    def parse(self, x):
        return x

    # 增删改查
    def add(self, x):
        raise NotImplementedError
    
    def delete(self, x):
        raise NotImplementedError
    
    def query(self, x):
        raise NotImplementedError
    
    def update(self, x):
        raise NotImplementedError