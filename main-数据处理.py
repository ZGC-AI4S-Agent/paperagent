import os
import json
import gradio as gr
import argparse  # 添加参数解析器
import torch
import concurrent.futures
import multiprocessing
import time
import numpy as np
from docagent.retrieval.embedding.gemini_embedding import VLLMQwenEmbedding
from docagent.retrieval.database.milvus_database import ChromaDatabase
from docagent.retrieval.retriever.simple_retriever import SimpleRetriever

# 修改时间处理函数
def process_publish_time(time_str):
    """处理不同格式的发布时间"""
    if not time_str or time_str == "Not Available":
        return None
    
    try:
        # 处理 ISO 格式时间 (如 "2022-11-21T19:10:33.302000Z")
        if 'T' in time_str:
            return int(time_str.split('T')[0].split('-')[0])
        
        # 处理简单日期格式 (如 "2025-01-02")
        if '-' in time_str:
            return int(time_str.split('-')[0])
        
        # 如果已经是整数，直接返回
        if isinstance(time_str, (int, float)):
            return int(time_str)
            
        return None
    except Exception as e:
        print(f"⚠️ 时间格式处理错误: {time_str}, 错误: {str(e)}")
        return None

# 修改作者处理函数
def process_authors(authors):
    """处理作者列表，清理格式"""
    if not authors:
        return ""
    
    if isinstance(authors, list):
        # 处理列表中的每个作者
        processed_authors = []
        for author in authors:
            if author and isinstance(author, str):
                # 移除多余的大括号和空格
                author = author.strip().strip('{}').strip()
                if author:
                    # 如果作者字符串中包含 "and"，则分割
                    if " and " in author:
                        and_authors = [a.strip() for a in author.split(" and ") if a.strip()]
                        processed_authors.extend(and_authors)
                    else:
                        processed_authors.append(author)
        return ", ".join(processed_authors)
    elif isinstance(authors, str):
        # 处理字符串形式的作者列表
        authors = authors.strip()
        if " and " in authors:
            # 分割并处理 "and" 分隔的作者
            and_authors = [a.strip() for a in authors.split(" and ") if a.strip()]
            return ", ".join(and_authors)
        return authors
    return ""

def process_paper(paper):
    """处理单篇论文数据"""
    try:
        # 处理字段名称映射
        if "abstract" in paper:
            paper["summary"] = paper.pop("abstract")
        if "journal_name" in paper:
            paper["venue"] = paper.pop("journal_name")
        if "publish_time" in paper:
            paper["published"] = paper.pop("publish_time")
        
        # 处理作者列表
        if "authors" in paper:
            paper["authors"] = process_authors(paper["authors"])
        
        # 处理发布时间
        if "published" in paper:
            processed_time = process_publish_time(paper["published"])
            if processed_time is not None:
                paper["published"] = processed_time
            else:
                return None
        
        # 确保必要字段存在
        if not all(key in paper for key in ["title", "authors", "summary"]):
            return None
        
        # 处理期刊名称
        if "venue" not in paper:
            paper["venue"] = "未知"
        elif not paper["venue"]:
            paper["venue"] = "未知"
        elif isinstance(paper["venue"], str):
            paper["venue"] = paper["venue"].strip()
            if paper["venue"] == "":
                paper["venue"] = "未知"
        
        return paper
    except Exception as e:
        print(f"⚠️ 处理论文数据时出错: {str(e)}")
        return None

# 系统初始化函数 - 多GPU并行处理
def initialize_system(data_dir="/home/dataset-assist-0/data/paperagent/data", reset_db=False, gpu_count=8, data_parallel_rank=0, data_parallel_size=1):
    """系统初始化函数，从指定文件夹加载所有JSON文件，利用多GPU并行处理
    
    Parameters:
    -----------
    data_dir: str
        数据目录路径
    reset_db: bool
        是否重置数据库
    gpu_count: int
        可用GPU数量
    data_parallel_rank: int
        数据并行组的排名（0或1）
    data_parallel_size: int
        数据并行组的数量（通常为2）
    """
    try:
        start_time = time.time()
        
        # 初始化嵌入模型
        tensor_parallel_size = 4  # 使用4个GPU做张量并行
        print(f"⏳ 初始化嵌入模型 (tensor_parallel_size={tensor_parallel_size})...")
        embedding = VLLMQwenEmbedding(tensor_parallel_size=tensor_parallel_size)
        print("✅ 嵌入模型初始化完成")

        # 初始化数据库和检索器
        collection_name = f"papers0520_dp{data_parallel_rank}" if data_parallel_size > 1 else "papers0520"
        print(f"💾 使用集合: {collection_name}")
        database = ChromaDatabase(collection_name=collection_name, dim=embedding.embedding_dim)
        
        # 如果需要重置数据库，则删除集合重新创建
        if reset_db:
            try:
                database.client.delete_collection("papers0520")
                database.collection = database.client.create_collection(
                    name="papers0520",
                    metadata={"hnsw:space": "cosine", "dimension": embedding.embedding_dim}
                )
                database.doc_count = 0
                print("✅ 数据库已重置")
            except Exception as e:
                print(f"⚠️ 重置数据库失败: {str(e)}")
        
        retriever = SimpleRetriever(embedding, database)
        print("✅ 数据库和检索器初始化完成")
        
        # 检查数据目录是否存在
        if not os.path.exists(data_dir):
            print(f"⚠️ 数据目录 {data_dir} 不存在，系统将以空数据库启动")
            return retriever
        
        # 获取所有JSON文件
        json_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]
        json_files.sort()  # 按文件名排序
        total_files = len(json_files)
        
        if total_files == 0:
            print("⚠️ 未找到任何JSON文件，系统将以空数据库启动")
            return retriever
            
        # 分割文件列表实现数据并行
        if data_parallel_size > 1:
            files_per_group = total_files // data_parallel_size
            start_idx = data_parallel_rank * files_per_group
            end_idx = start_idx + files_per_group if data_parallel_rank < data_parallel_size - 1 else total_files
            json_files = json_files[start_idx:end_idx]
            print(f"📊 数据并行组 {data_parallel_rank+1}/{data_parallel_size}，处理 {len(json_files)}/{total_files} 个文件")
        
        print(f"⏳ 开始处理 {len(json_files)} 个文件...")
        total_papers = 0
        file_paths = [os.path.join(data_dir, filename) for filename in json_files]
        processed_files = 0
        
        # 创建进程池，处理每个文件
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            for batch_start in range(0, len(file_paths), 10):  # 每次处理10个文件
                batch_end = min(batch_start + 10, len(file_paths))
                batch_files = file_paths[batch_start:batch_end]
                
                # 并行处理每个文件
                file_papers_list = []
                for file_path in batch_files:
                    try:
                        # 读取JSON文件
                        with open(file_path, 'r', encoding='utf-8') as f:
                            papers = json.load(f)
                        
                        # 处理论文数据
                        file_papers = []
                        if isinstance(papers, list):
                            paper_results = list(executor.map(process_paper, papers))
                            file_papers = [p for p in paper_results if p]
                        elif isinstance(papers, dict):
                            processed_paper = process_paper(papers)
                            if processed_paper:
                                file_papers.append(processed_paper)
                        
                        file_papers_list.extend(file_papers)
                    except Exception as e:
                        print(f"❌ 处理文件出错: {os.path.basename(file_path)}")
                
                processed_files += len(batch_files)
                progress = processed_files / len(file_paths) * 100
                print(f"🔄 已处理: {processed_files}/{len(file_paths)} 个文件 ({progress:.1f}%)")
                
                # 处理完一批文件后生成嵌入并导入数据库
                if file_papers_list:
                    # 分批处理嵌入，提高批处理大小以提升GPU利用率
                    for i in range(0, len(file_papers_list), 128):
                        batch_papers = file_papers_list[i:i+128]
                        # 使用检索器添加文档
                        retriever.add_batched_documents(batch_papers, batch_size=128)
                    
                    total_papers += len(file_papers_list)
                    print(f"📝 总计已导入: {total_papers} 篇论文")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if total_papers > 0:
            print(f"\n✅ 处理完成: 导入 {total_papers} 篇论文，用时 {processing_time:.2f} 秒，平均每文件 {processing_time/len(json_files):.2f} 秒")
        else:
            print("\n⚠️ 未找到有效的论文数据，系统将以空数据库启动")
        
        return retriever

    except Exception as e:
        print(f"❌ 系统初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 修改构建Milvus过滤表达式函数
def build_filters(journal=None, min_year=None, max_year=None, author=None):
    """构建Milvus过滤表达式"""
    filters = []
    if journal and journal.strip():
        filters.append(f'venue == "{journal}"')
    
    time_filters = []
    if min_year and min_year != "":
        time_filters.append(f"published >= {int(min_year)}")
    if max_year and max_year != "":
        time_filters.append(f"published <= {int(max_year)}")
    if time_filters:
        filters.append("(" + " AND ".join(time_filters) + ")")
        
    # 修改作者匹配逻辑，支持模糊匹配单个作者
    if author and author.strip():
        # 移除多余的空格并分割作者名
        author_terms = [name.strip() for name in author.split(',')]
        author_filters = []
        for term in author_terms:
            if term:  # 确保不是空字符串
                # 使用 LIKE 进行模糊匹配，不区分大小写
                author_filters.append(f'authors LIKE "%{term}%"')
        if author_filters:
            # 使用 OR 连接多个作者条件，匹配任意一个作者即可
            filters.append("(" + " OR ".join(author_filters) + ")")
            
    return " AND ".join(filters) if filters else ""

# 修改核心检索函数
def search_papers(query_title, query_abstract, top_k=5, journal=None, min_year=None, max_year=None, author=None):
    """核心检索函数"""
    # 修改输入验证逻辑，允许仅通过作者检索
    if not (query_title or query_abstract or author):
        return "<div class='output-container'><div style='text-align:center;color:#666;'>⚠️ 请至少输入标题、摘要或作者名称</div></div>"
    
    # 构建查询文本
    query_parts = []
    if query_title:
        query_parts.append(f"Title: {query_title}")
    if query_abstract:
        query_parts.append(f"Abstract: {query_abstract}")
    if author:
        query_parts.append(f"Author: {author}")
    
    query_text = "\n".join(query_parts)
    
    filter_expr = build_filters(journal, min_year, max_year, author)
    try:
        results = retriever.retrieve(query_text=query_text, top_k=top_k, filter_expression=filter_expr)
    except Exception as e:
        return f"<div class='output-container'><div style='text-align:center;color:#666;'>❌ 检索失败: {str(e)}</div></div>"
    
    if not results:
        return "<div class='output-container'><div style='text-align:center;color:#666;'>🔍 未找到相关论文</div></div>"

    # 开始构建 HTML 输出，确保包裹在 .output-container 中
    html_output = ["<div class='output-container'>"]
    for idx, paper in enumerate(results, 1):
        summary_snippet = paper['entity']['summary'][:300] + "..." if len(paper['entity']['summary']) > 300 else \
        paper['entity']['summary']
        link = paper['entity'].get('link', '无链接')
        html_output.append(f"""
            <div class="paper-result">
                <h3>匹配结果 #{idx}</h3>
                <p><strong>📰 期刊:</strong> {paper['entity'].get('venue', '未知')}</p>
                <p><strong>📅 年份:</strong> {paper['entity'].get('published', '未知')}</p>
                <p><strong>📖 标题:</strong> {paper['entity']['title']}</p>
                <p><strong>👥 作者:</strong> {paper['entity']['authors']}</p>
                <p><strong>📄 摘要:</strong> {summary_snippet}</p>
                <p><strong>🔗 链接:</strong> <a href="{link}" target="_blank">{link}</a></p>
            </div>
            """)
    html_output.append("</div>")
    return "".join(html_output)

# 修改统计函数的参数定义
def analyze_authors_publications(keyword1, keyword2, keyword3, keyword4, keyword5, keyword6, min_year=None, max_year=None):
    """统计作者在特定领域的论文发表数量"""
    try:
        # 将所有关键词放入列表并清理
        keywords_list = [keyword1, keyword2, keyword3, keyword4, keyword5, keyword6]
        keywords = [kw.strip() for kw in keywords_list if kw.strip()]
        if not keywords:
            return "<div class='output-container'><div style='text-align:center;color:#666;'>⚠️ 请至少输入一个关键词</div></div>"
            
        # 存储所有检索到的论文，使用字典存储完整论文信息
        papers_dict = {}
        
        # 构建基础过滤条件
        filter_expr = build_filters(min_year=min_year, max_year=max_year)
        # 添加排除 arxiv 的条件
        if filter_expr:
            filter_expr += ' AND venue != "nature"'
        else:
            filter_expr = 'venue != "nature"'
            
        # 对每个关键词进行检索
        for keyword in keywords:
            try:
                results = retriever.retrieve(query_text=keyword, top_k=200, filter_expression=filter_expr)
                # 存储完整的论文信息
                for paper in results:
                    title = paper['entity']['title']
                    if title not in papers_dict:  # 避免重复添加
                        papers_dict[title] = paper['entity']
            except Exception as e:
                print(f"检索关键词 '{keyword}' 时出错: {str(e)}")
                continue
                
        if not papers_dict:
            return "<div class='output-container'><div style='text-align:center;color:#666;'>🔍 未找到相关论文</div></div>"
        
        # 统计作者发表数量
        author_stats = {}
        # 直接使用存储的论文信息进行统计
        for paper in papers_dict.values():
            authors = paper['authors'].split(", ")
            for author in authors:
                author = author.strip()
                if author:
                    author_stats[author] = author_stats.get(author, 0) + 1
        
        # 排序作者按论文数量
        sorted_authors = sorted(author_stats.items(), key=lambda x: x[1], reverse=True)
        
        # 构建HTML输出
        html_output = ["<div class='output-container'>"]
        html_output.append(f"<h2>研究领域作者论文发表统计</h2>")
        html_output.append(f"<p>检索关键词: {', '.join(keywords)}</p>")
        html_output.append(f"<p>总共找到相关论文: {len(papers_dict)} 篇</p>")
        html_output.append("<div class='stats-container'>")
        
        for idx, (author, count) in enumerate(sorted_authors[:30], 1):  # 显示前30名
            html_output.append(f"""
                <div class="author-stat-card">
                    <h3>#{idx} {author}</h3>
                    <p class="paper-count">发表论文数：{count}</p>
                </div>
            """)
        
        html_output.append("</div></div>")
        return "".join(html_output)
    
    except Exception as e:
        return f"<div class='output-container'><div style='text-align:center;color:#666;'>❌ 统计失败: {str(e)}</div></div>"

# CSS样式（保持不变）
css = """
/* 全局样式 */
.container {
    max-width: 1200px;
    margin: 0 auto;
}

/* 标题样式 */
h1 {
    text-align: center;
    color: #2c3e50;
    padding: 20px 0;
    margin-bottom: 30px;
    border-bottom: 2px solid #eee;
}

/* 输入面板样式 */
.input-container {
    background: #ffffff;
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}

/* 输入框样式 */
.gr-textbox, .gr-textarea {
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    padding: 10px !important;
    font-size: 14px !important;
    transition: all 0.3s ease;
}

.gr-textbox:focus, .gr-textarea:focus {
    border-color: #2196f3 !important;
    box-shadow: 0 0 0 2px rgba(33,150,243,0.1) !important;
}

/* 按钮样式 */
.gr-button {
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.gr-button.primary {
    background: #2196f3 !important;
    border: none !important;
}

.gr-button.primary:hover {
    background: #1976d2 !important;
    transform: translateY(-1px);
}

/* 结果面板样式 */
.output-container {
    background: #ffffff;
    border-radius: 10px;
    padding: 20px !important;
    height: 800px !important;
    overflow-y: auto;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
}

.paper-result {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 15px !important;
    margin: 15px 0 !important;
    border: 1px solid #e9ecef;
    transition: all 0.3s ease;
}

.paper-result:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}

.paper-result h3 {
    color: #1976d2;
    margin: 0 0 15px 0 !important;
    font-size: 18px !important;
}

.paper-result p {
    margin: 8px 0 !important;
    line-height: 1.5;
}

/* 分组样式 */
.search-group {
    background: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
}

.search-group-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 15px;
    color: #2c3e50;
}

/* 统计结果样式 */
.stats-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 15px;
    padding: 15px;
}

.author-stat-card {
    background: #ffffff;
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: all 0.3s ease;
}

.author-stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.author-stat-card h3 {
    color: #2196f3;
    margin: 0 0 10px 0;
    font-size: 16px;
}

.paper-count {
    font-size: 14px;
    color: #666;
    margin: 0;
}

/* 标签页样式 */
.tab-selected {
    background: #2196f3 !important;
    color: white !important;
}
"""

# 示例期刊和年份列表 - 移到这里，确保在构建界面前定义
journal_list = ["nature", "science", "cell"]
year_list = [str(year) for year in range(2025, 1899, -1)]

# 更新Gradio界面
with gr.Blocks(title="AI4s学术论文智能检索平台", theme=gr.themes.Soft(), css=css) as interface:
    gr.Markdown("""
        # 📚 AI4s学术论文智能检索平台
        ### 智能检索您需要的学术论文
    """)
    
    # 创建标签页
    with gr.Tabs():
        # 论文检索标签页
        with gr.Tab("论文检索"):
            with gr.Row(equal_height=True):
                # 左侧输入面板
                with gr.Column(scale=1, min_width=300):
                    with gr.Column(elem_classes="input-container"):
                        # 基础搜索区域
                        with gr.Column(elem_classes="search-group"):
                            gr.Markdown("### 📝 基础搜索", elem_classes="search-group-title")
                            title_input = gr.Textbox(
                                label="论文标题",
                                placeholder="输入论文标题（选填）...",
                            )
                            abstract_input = gr.TextArea(
                                label="论文摘要",
                                placeholder="输入论文摘要（选填）...",
                                lines=4,
                            )
                        
                        # 高级筛选区域
                        with gr.Accordion("🔍 高级筛选", open=True):
                            with gr.Column(elem_classes="search-group"):
                                author_input = gr.Textbox(
                                    label="作者姓名",
                                    placeholder="输入作者姓名，支持模糊匹配...",
                                )
                                journal_input = gr.Dropdown(
                                    label="目标期刊",
                                    choices=journal_list,
                                    value=None,
                                )
                                with gr.Row():
                                    min_year_input = gr.Dropdown(
                                        label="起始年份",
                                        choices=year_list,
                                        value=None
                                    )
                                    max_year_input = gr.Dropdown(
                                        label="结束年份",
                                        choices=year_list,
                                        value=None
                                    )
                        
                        # 搜索控制区域
                        with gr.Row():
                            top_k_input = gr.Slider(
                                minimum=1,
                                maximum=30,
                                value=3,
                                step=1,
                                label="显示论文数量"
                            )
                            search_btn = gr.Button(
                                "开始检索",
                                variant="primary",
                                scale=1
                            )

                # 右侧结果面板
                with gr.Column(scale=2):
                    output_panel = gr.HTML(
                        value="<div class='output-container'><div style='text-align:center;color:#666;padding:20px;'>等待检索，请输入搜索条件...</div></div>"
                    )

        # 作者统计标签页
        with gr.Tab("作者统计"):
            with gr.Column():
                with gr.Column(elem_classes="input-container"):
                    # 修改为6个关键词输入框
                    keywords_inputs = []
                    for i in range(6):
                        keywords_inputs.append(
                            gr.Textbox(
                                label=f"研究领域关键词 {i+1}",
                                placeholder=f"输入第{i+1}个研究领域关键词...",
                                value="" if i > 0 else "ai for science"
                            )
                        )
                    with gr.Row():
                        stat_min_year = gr.Dropdown(
                            label="起始年份",
                            choices=year_list,
                            value=None
                        )
                        stat_max_year = gr.Dropdown(
                            label="结束年份",
                            choices=year_list,
                            value=None
                        )
                    analyze_btn = gr.Button("开始统计", variant="primary")
                
                stats_output = gr.HTML(
                    value="<div class='output-container'><div style='text-align:center;color:#666;'>等待统计...</div></div>"
                )
    
    # 事件绑定
    search_btn.click(
        fn=search_papers,
        inputs=[title_input, abstract_input, top_k_input, journal_input, min_year_input, max_year_input, author_input],
        outputs=output_panel
    )
    
    analyze_btn.click(
        fn=analyze_authors_publications,
        inputs=[*keywords_inputs, stat_min_year, stat_max_year],
        outputs=stats_output
    )

if __name__ == "__main__":
    # 添加命令行参数
    parser = argparse.ArgumentParser(description="AI4s学术论文智能检索平台")
    parser.add_argument('--reset', action='store_true', help='重置数据库并重新导入所有数据')
    parser.add_argument('--port', type=int, default=8081, help='服务端口号')
    parser.add_argument('--no-share', action='store_true', help='不创建公共链接')
    parser.add_argument('--gpu-count', type=int, default=8, help='使用的GPU数量')
    parser.add_argument('--data-dir', type=str, default="/home/dataset-assist-0/data/paperagent/data", help='数据目录路径')
    parser.add_argument('--dp-rank', type=int, default=0, help='数据并行组编号(0或1)')
    parser.add_argument('--dp-size', type=int, default=1, help='数据并行组数量(通常为2)')
    parser.add_argument('--merge', action='store_true', help='合并所有数据并行集合到主集合')
    args = parser.parse_args()
    
    # 处理数据合并请求
    if args.merge and args.dp_size > 1:
        try:
            import chromadb
            import time
            
            # 生成唯一的合并操作ID（使用时间戳）
            merge_id = int(time.time())
            
            print("🔄 开始合并集合...")
            print(f"🆔 本次合并操作ID: {merge_id}")
            client = chromadb.PersistentClient(path="/home/dataset-assist-0/data/chromadb")
            print(f"📂 使用数据库路径: /home/dataset-assist-0/data/chromadb")
            
            # 列出所有现有集合进行检查
            existing_collections = client.list_collections()
            print(f"📋 现有集合列表:")
            for coll in existing_collections:
                print(f"  - {coll.name} (ID: {coll.id})")
            
            # 检查并获取主集合 - 不使用自定义嵌入函数，保持与源集合一致
            main_collection_name = "papers0520"
            try:
                main_collection = client.get_collection(name=main_collection_name)
                original_count = main_collection.count()
                print(f"📊 主集合 {main_collection_name} 中已有 {original_count} 条记录")
            except:
                main_collection = client.create_collection(name=main_collection_name)
                original_count = 0
                print(f"✅ 已创建主集合 {main_collection_name}")
            
            # 记录总合并文档数
            total_merged_docs = 0
            
            # 合并所有数据并行集合的数据
            for dp_rank in range(args.dp_size):
                src_collection_name = f"papers0520_dp{dp_rank}"
                try:
                    # 不指定嵌入函数，使用集合原有的嵌入函数
                    src_collection = client.get_collection(name=src_collection_name)
                    doc_count = src_collection.count()
                    if doc_count == 0:
                        print(f"⚠️ 集合 {src_collection_name} 为空，跳过")
                        continue
                        
                    print(f"📤 从集合 {src_collection_name} 导出 {doc_count} 条记录...")
                    
                    # 分批获取并导入数据
                    batch_size = 1000
                    for i in range(0, doc_count, batch_size):
                        # 获取当前批次的文档和嵌入
                        results = src_collection.get(
                            limit=batch_size,
                            offset=i,
                            include=["documents", "embeddings", "metadatas"]
                        )
                        
                        if not results["ids"]:
                            continue
                            
                        # 修改ID，添加合并ID和来源前缀，确保多次合并也不会重复
                        modified_ids = [f"merge{merge_id}_dp{dp_rank}_{id}" for id in results["ids"]]
                        
                        # 添加到主集合
                        main_collection.add(
                            ids=modified_ids,
                            embeddings=results["embeddings"],
                            documents=results["documents"],
                            metadatas=results["metadatas"]
                        )
                        
                        # 更新合并计数
                        batch_count = len(modified_ids)
                        total_merged_docs += batch_count
                        
                        print(f"✅ 已处理 {min(i+batch_size, doc_count)}/{doc_count} 条记录")
                    
                    print(f"✅ 已完成集合 {src_collection_name} 的合并")
                except Exception as e:
                    print(f"❌ 合并集合 {src_collection_name} 失败: {str(e)}")
            
            final_count = main_collection.count()
            print(f"📊 合并前主集合中有 {original_count} 条记录")
            print(f"📊 合并后主集合 {main_collection_name} 中有 {final_count} 条记录")
            print(f"📈 本次合并添加了 {final_count - original_count} 条记录")
            print(f"📈 理论上应添加 {total_merged_docs} 条记录")
            print("✅ 集合合并完成")
            
            # 删除源集合
            print("🗑️ 开始删除源集合...")
            for dp_rank in range(args.dp_size):
                src_collection_name = f"papers0520_dp{dp_rank}"
                try:
                    client.delete_collection(src_collection_name)
                    print(f"✅ 已删除源集合 {src_collection_name}")
                except Exception as e:
                    print(f"⚠️ 删除集合 {src_collection_name} 失败: {str(e)}")
            
            print("✅ 所有操作完成")
            exit(0)
        except Exception as e:
            print(f"❌ 合并集合失败: {str(e)}")
            import traceback
            traceback.print_exc()
            exit(1)
    
    # 设置可见GPU - 根据数据并行组编号设置
    if args.dp_size > 1:
        gpu_per_group = args.gpu_count // args.dp_size
        visible_gpus = list(range(args.dp_rank * gpu_per_group, (args.dp_rank + 1) * gpu_per_group))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, visible_gpus))
        print(f"🔄 数据并行组 {args.dp_rank+1}/{args.dp_size}，使用 GPU {visible_gpus}")
    
    # 系统初始化
    print(f"🚀 系统启动 - 端口: {args.port}")
    retriever = initialize_system(
        data_dir=args.data_dir, 
        reset_db=args.reset, 
        gpu_count=args.gpu_count // args.dp_size if args.dp_size > 1 else args.gpu_count,
        data_parallel_rank=args.dp_rank,
        data_parallel_size=args.dp_size
    )
    
    # 启动界面
    interface.launch(server_port=args.port, share=not args.no_share)