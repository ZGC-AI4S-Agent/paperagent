# simple_retriever.py - 检索器实现
from tqdm import tqdm


class SimpleRetriever:
    """检索器实现"""

    def __init__(self, embedding_model, database):
        self.embedder = embedding_model
        self.db = database

    def add_batched_documents(self, documents, batch_size=64):
        """批量添加文档"""
        for i in tqdm(range(0, len(documents), batch_size), desc="插入数据"):
            batch = documents[i:i + batch_size]
            # 将标题和摘要合并
            combined_texts = [f"{doc['title']} {doc['summary']}" for doc in batch]
            # 使用通用的 embed 方法
            embeddings = self.embedder.embed(combined_texts)

            prepared_batch = [{
                **doc,
                "vector": emb
            } for doc, emb in zip(batch, embeddings)]

            self.db.insert_documents(prepared_batch)

    def retrieve(self, query_text, top_k=5, filter_expression=None):
        """执行检索"""
        # 检查 query_text 是否为空
        if not query_text:
            print("⚠️ 检索文本为空，无法执行检索。")
            return []
            
        # 调用 embed 方法，它接收一个列表并返回一个列表
        # 因此，即使只有一个查询文本，也要传入列表，并取结果列表的第一个元素
        query_vector_list = self.embedder.embed([query_text])
        
        # 确保返回了向量
        if not query_vector_list:
             print(f"❌ 无法为查询文本生成嵌入向量: '{query_text}'")
             return []
             
        query_vector = query_vector_list[0]
        
        return self.db.similarity_search(
            query_vector=query_vector,
            top_k=top_k,
            filter_expression=filter_expression
        )