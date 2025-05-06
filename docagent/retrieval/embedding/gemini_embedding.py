import os
import numpy as np
# 移除 SentenceTransformer 导入
# from sentence_transformers import SentenceTransformer
from vllm import LLM # 导入 vLLM
# 移除 OpenAI 相关导入和代理设置

from docagent.retrieval.embedding.base import BaseEmbedding

# 重命名类以反映新的模型和框架
class VLLMQwenEmbedding(BaseEmbedding):
    def __init__(self, model_name='/home/dataset-assist-0/data/paperagentui/models/models--Alibaba-NLP--gte-Qwen2-7B-instruct/snapshots/a8d08b36ada9cacfe34c4d6f80957772a025daf2', tensor_parallel_size=1, trust_remote_code=True, **kwargs):
        """
        使用 vLLM 初始化 Qwen 嵌入模型 (从项目内指定路径加载)。

        Args:
            model_name (str): 项目内模型快照文件夹的绝对路径。
            tensor_parallel_size (int): 用于张量并行的大小。
            trust_remote_code (bool): 是否信任远程代码（对于某些模型是必需的）。
            **kwargs: 其他传递给 vLLM 的参数。
        """
        super().__init__()
        try:
            # 检查本地路径是否存在
            if not os.path.isdir(model_name):
                raise FileNotFoundError(
                    f"项目内模型路径 '{model_name}' 不存在或不是一个文件夹。"
                    f"请确认模型是否已成功复制到此路径，并且路径正确。"
                )
            # 进一步检查 config.json 是否存在
            config_path = os.path.join(model_name, 'config.json')
            if not os.path.isfile(config_path):
                 raise FileNotFoundError(
                    f"在指定的模型路径 '{model_name}' 下未找到 'config.json' 文件。"
                    f"请确认模型复制是否完整。"
                 )

            # 使用 vLLM 加载项目内复制的模型
            print(f"⏳ 正在使用 vLLM 从项目路径加载模型 {model_name}...")
            self.model = LLM(
                model=model_name, # 直接使用项目内的绝对路径
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                **kwargs
            )
            print(f"✅ 成功使用 vLLM 加载模型 {model_name}")

            # 动态获取嵌入维度
            print("⚙️ 正在获取嵌入维度...")
            dummy_text = "获取维度测试文本"
            # 使用 vLLM 的 encode 方法获取嵌入
            dummy_output = self.model.encode([dummy_text]) # 使用列表作为输入

            # 检查返回结果是否有效
            if not dummy_output or not hasattr(dummy_output[0], 'outputs') or not hasattr(dummy_output[0].outputs, 'embedding'):
                 raise ValueError("无法从模型输出中获取嵌入向量，请检查 vLLM 版本或模型兼容性。")

            self.embedding_dim = len(dummy_output[0].outputs.embedding)
            print(f"✅ 嵌入维度确定为: {self.embedding_dim}")
            print(f"⚠️ 警告：请确保 Milvus 数据库的维度设置为 {self.embedding_dim}")

        except ImportError:
             print("❌ 错误: 未找到 vllm 库。请确保已安装 vLLM: `pip install vllm`")
             raise
        except Exception as e:
            print(f"❌ 使用 vLLM 加载模型 {model_name} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise # 抛出异常，因为没有模型无法继续

        # # 保留代理设置的清理（如果之前设置过） - 加载本地模型时通常不需要删除代理设置
        # if "HTTP_PROXY" in os.environ:
        #     del os.environ["HTTP_PROXY"]
        #     print("移除了 HTTP_PROXY 环境变量")
        # if "HTTPS_PROXY" in os.environ:
        #     del os.environ["HTTPS_PROXY"]
        #     print("移除了 HTTPS_PROXY 环境变量")

    def embed(self, texts):
        """
        使用 vLLM 模型为文本列表生成嵌入。

        Args:
            texts (list or str): 需要嵌入的文本或文本列表。

        Returns:
            list: 嵌入向量列表 (每个向量是 list of floats)。
        """
        # 确保输入是列表
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, list):
            raise TypeError("输入必须是字符串或字符串列表")
        if not texts:
            return []

        try:
            # 使用 vLLM 的 encode 方法生成嵌入
            # 它期望一个文本列表作为输入
            request_outputs = self.model.encode(texts)

            # 从结果中提取嵌入向量
            embeddings = [output.outputs.embedding for output in request_outputs]

            # 检查嵌入是否为空或维度不匹配（可选，增加健壮性）
            if not embeddings or any(len(e) != self.embedding_dim for e in embeddings):
                print(f"❌ 嵌入结果异常：返回了空列表或维度不匹配 ({self.embedding_dim})。")
                 # 对于无效嵌入，可以返回零向量或抛出错误
                return [[0.0] * self.embedding_dim for _ in texts]

            # vLLM 返回的 embedding 已经是 list of floats
            return embeddings

        except Exception as e:
            print(f"❌ vLLM 嵌入处理错误: {str(e)}")
            import traceback
            traceback.print_exc()
            # 出错时返回对应维度的零向量列表
            return [[0.0] * self.embedding_dim for _ in texts]

# 移除旧的示例代码和注释
