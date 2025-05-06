# milvus_database.py
import chromadb
from chromadb.config import Settings
import os
import uuid  # 添加uuid模块导入
import re # 需要导入 re 模块

# Define the maximum number of author fields to store separately
MAX_AUTHORS_PER_PAPER = 50

class ChromaDatabase:
    def __init__(self, collection_name, dim):
        """初始化数据库连接"""
        # 设置存储路径 - 使用特定实例目录
        # 可以选择使用根目录或特定实例目录
        persist_directory = "/home/dataset-assist-0/data/chromadb"
        
        # 确保目录存在
        os.makedirs(persist_directory, exist_ok=True)
        
        # 初始化 ChromaDB 客户端
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        self.collection_name = collection_name
        self.dim = dim
        
        # 创建或获取集合
        # Check if the collection exists to potentially update metadata definition if needed
        # Although ChromaDB is flexible, explicitly defining metadata structure might be good practice
        # For now, we rely on dynamic schema
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"✅ 成功获取已存在的集合: {collection_name}")
        except Exception: # Handle cases where collection might not exist or other errors
             print(f"ℹ️ 集合 {collection_name} 不存在或获取失败，将创建新集合。")
             # Ensure dimension is passed correctly if creating
             # Metadata for space/dimension is often set at creation, let's keep it
             collection_metadata = {"hnsw:space": "cosine"}
             # Note: ChromaDB schema can be dynamic, adding dimension here might be informational
             # collection_metadata["dimension"] = dim # Re-check if needed/supported here

             self.collection = self.client.get_or_create_collection(
                 name=collection_name,
                 metadata=collection_metadata
             )
             print(f"✅ 成功创建新集合: {collection_name}")


        # 记录集合当前文档数，用于生成唯一ID
        self.doc_count = self.collection.count()
        print(f"当前集合中已有文档数: {self.doc_count}")

    def insert_documents(self, documents):
        """批量插入文档，并将作者拆分到单独字段"""
        # 准备数据
        ids = []
        documents_list = []
        embeddings = []
        metadatas = []

        for i, doc in enumerate(documents):
            unique_id = f"doc_{self.doc_count + i}"
            ids.append(unique_id)
            documents_list.append(f"{doc['title']} {doc['summary']}") # Document content remains the same
            embeddings.append(doc['vector'])

            # Prepare metadata, including split authors
            metadata = {
                "title": doc["title"],
                "summary": doc["summary"],
                "authors": doc["authors"], # Keep the original string/list for display
                "venue": doc["venue"],
                "link": doc.get("link", ""), # <-- Use 'link' from source, store as 'link'
                "published": doc["published"]
            }

            # Split authors and add author1, author2, ... fields
            authors_list = []
            if isinstance(doc["authors"], list):
                 authors_list = doc["authors"]
            elif isinstance(doc["authors"], str):
                 # Simple split, adjust if delimiter is different or needs trimming
                 authors_list = [a.strip() for a in doc["authors"].split(',') if a.strip()]

            for j in range(MAX_AUTHORS_PER_PAPER):
                author_field_name = f"author{j+1}"
                if j < len(authors_list):
                    metadata[author_field_name] = authors_list[j]
                else:
                    # Set remaining author fields to a default value (e.g., empty string or None)
                    # Using empty string might be safer for filtering ($eq: "")
                    metadata[author_field_name] = ""

            metadatas.append(metadata)

        # 批量插入
        try:
            self.collection.add(
                ids=ids,
                documents=documents_list, # Content for vector search
                embeddings=embeddings,
                metadatas=metadatas # Metadata including author1..N
            )
            self.doc_count += len(documents)
            print(f"✅ 成功插入 {len(documents)} 条数据 (含拆分作者字段)，当前总数: {self.doc_count}")
        except Exception as e:
            print(f"❌ 数据插入失败: {str(e)}")
            # Consider logging the problematic batch/metadata for debugging
            # import traceback
            # traceback.print_exc()


    def similarity_search(self, query_vector, top_k=5, filter_expression=None):
        """相似性搜索，直接使用传入的 filter_expression 作为 where 条件。"""
        # 注意：filter_expression 现在应该是一个 ChromaDB where document (字典)
        # 移除了之前的字符串解析逻辑
        where_conditions = filter_expression # Directly use the dictionary

        # 添加调试打印，确认传入的条件
        print(f"[Debug DB] Received where_conditions for query: {where_conditions}")

        try:
            # Execute DB search with the provided where clause
            print(f"🔍 执行 ChromaDB 查询，Top K: {top_k}, DB Where: {where_conditions}")
            results = self.collection.query(
                query_embeddings=[query_vector] if query_vector else None,
                n_results=top_k,
                where=where_conditions, # Use the received dictionary directly
                include=["metadatas", "distances"]
            )

            # Format results (no client-side filtering needed now)
            formatted_results = []
            if results and results["ids"] and results["ids"][0]:
                for i in range(len(results["ids"][0])):
                    if results["metadatas"] and len(results["metadatas"]) > 0 and results["metadatas"][0] and len(results["metadatas"][0]) > i:
                        metadata = results["metadatas"][0][i]
                        # Remove author1..N fields before returning to keep API clean? Optional.
                        # entity = {k: v for k, v in metadata.items() if not k.startswith('author')}
                        entity = metadata # Keep all metadata for now

                        formatted_results.append({
                            "entity": entity,
                             # Handle case where distances might be None if query_embeddings wasn't provided/used effectively
                            "distance": results["distances"][0][i] if results["distances"] and results["distances"][0] else None
                        })
                    else:
                        print(f"⚠️ 警告：查询结果中缺少索引 {i} 的元数据。")
            else:
                 print("ℹ️ ChromaDB 查询未返回任何结果。")


            return formatted_results

        except Exception as e:
            print(f"❌ ChromaDB 搜索失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return []