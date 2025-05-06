import os
import gradio as gr
import argparse
import torch
import time
import chromadb
from chromadb.config import Settings
from docagent.retrieval.embedding.gemini_embedding import VLLMQwenEmbedding
from docagent.retrieval.database.milvus_database import ChromaDatabase
from docagent.retrieval.retriever.simple_retriever import SimpleRetriever

# Define the core JS logic as a string (to be embedded in onclick)
# Note: This string itself should not contain the outer function definition or call parenthesis.
# We also need to be careful with quotes and newlines within this string.
js_onclick_logic = r"""
    console.log('Inline onclick logic started for author:', authorName);
    /* Helper function definition */
    function findInputElementByLabelText(labelText) {
        let labelSpan = Array.from(document.querySelectorAll('label span')).find(el => el.textContent.includes(labelText));
        if (labelSpan) {
            let container = labelSpan.closest('.gradio-textbox, .gradio-textarea, .gradio-dropdown, .block');
            if (container) {
                let inputElement = container.querySelector('textarea, input[type="text"], input[type="search"], select');
                let clearButton = container.querySelector('.clear-button, .clear');
                return { input: inputElement, clearBtn: clearButton, container: container };
            }
        }
        console.warn(`Could not find container or input for label: ${labelText}`);
        return { input: null, clearBtn: null, container: null };
    }
    /* Clear other fields */
    console.log('Clearing other fields...');
    let titleInfo = findInputElementByLabelText('论文标题');
    if (titleInfo.input) { titleInfo.input.value = ''; titleInfo.input.dispatchEvent(new Event('input', { bubbles: true })); } else { console.warn('Title input not found'); }
    let abstractInfo = findInputElementByLabelText('论文摘要');
    if (abstractInfo.input && abstractInfo.input.tagName === 'TEXTAREA') { abstractInfo.input.value = ''; abstractInfo.input.dispatchEvent(new Event('input', { bubbles: true })); } else { console.warn('Abstract textarea not found'); }
    let journalInfo = findInputElementByLabelText('目标期刊');
    if (journalInfo.clearBtn) { journalInfo.clearBtn.click(); } else if (journalInfo.input) { journalInfo.input.value = ''; journalInfo.input.dispatchEvent(new Event('change', { bubbles: true })); } else { console.warn('Journal dropdown not found'); }
    let minYearInfo = findInputElementByLabelText('起始年份');
    if (minYearInfo.clearBtn) { minYearInfo.clearBtn.click(); } else if (minYearInfo.input) { minYearInfo.input.value = ''; minYearInfo.input.dispatchEvent(new Event('change', { bubbles: true })); } else { console.warn('Min year dropdown not found'); }
    let maxYearInfo = findInputElementByLabelText('结束年份');
    if (maxYearInfo.clearBtn) { maxYearInfo.clearBtn.click(); } else if (maxYearInfo.input) { maxYearInfo.input.value = ''; maxYearInfo.input.dispatchEvent(new Event('change', { bubbles: true })); } else { console.warn('Max year dropdown not found'); }
    console.log('Finished clearing fields.');
    /* Set Author and Trigger Search */
    let authorInfo = findInputElementByLabelText('作者姓名');
    if (!authorInfo.input) {
        console.error('Could not find the author input element.');
        alert('无法找到作者输入框，无法自动搜索。请手动复制作者名。');
        return;
    }
    let searchButton = Array.from(document.querySelectorAll('button')).find(el => el.textContent.trim() === '开始检索');
    if (!searchButton) {
        console.error('Could not find the search button.');
        alert('无法找到搜索按钮，无法自动搜索。');
        return;
    }
    console.log('Setting author input:', authorInfo.input);
    authorInfo.input.value = authorName;
    authorInfo.input.dispatchEvent(new Event('input', { bubbles: true }));
    console.log('Clicking search button:', searchButton);
    searchButton.click();
    console.log('Author set and search button clicked.');
"""

# 构建过滤表达式函数
def build_filters(journal=None, min_year=None, max_year=None, author=None):
    """构建适用于 ChromaDB 的过滤表达式 (where document)"""
    # 注意：这个函数现在返回 ChromaDB 的 where dict，而不是字符串
    db_conditions = [] # 使用列表存储 AND 条件

    # Venue Filter ($eq)
    if journal and journal.strip():
        db_conditions.append({"venue": {"$eq": journal.strip()}})

    # Time Filter ($gte, $lte, $and)
    time_conditions = []
    if min_year and min_year != "":
        try:
            time_conditions.append({"published": {"$gte": int(min_year)}})
        except ValueError:
            print(f"⚠️ 无效的最小年份值: {min_year}")
    if max_year and max_year != "":
        try:
            time_conditions.append({"published": {"$lte": int(max_year)}})
        except ValueError:
            print(f"⚠️ 无效的最大年份值: {max_year}")

    if len(time_conditions) > 1:
        db_conditions.append({"$and": time_conditions})
    elif len(time_conditions) == 1:
        db_conditions.append(time_conditions[0])

    # Author Filter ($or across author1..authorN)
    if author and author.strip():
        # 假设作者输入是单个姓名或逗号分隔的多个姓名
        # 这里我们处理单个作者精确匹配的情况，与milvus_database.py中的解析逻辑类似
        # 如果需要支持UI输入多个作者并进行OR/AND组合，逻辑会更复杂
        # 当前简化为：如果输入了作者名，则在 author1 到 author50 中查找完全匹配项
        author_name_to_match = author.strip() # 直接使用输入框的值
        author_or_conditions = []
        # 从 milvus_database.py 导入或重新定义 MAX_AUTHORS_PER_PAPER
        # 为简单起见，直接在这里用字面量 50
        MAX_AUTHORS_PER_PAPER = 50
        for i in range(1, MAX_AUTHORS_PER_PAPER + 1):
            author_or_conditions.append({f"author{i}": {"$eq": author_name_to_match}})

        if author_or_conditions:
             # 只需要一个 $or 条件来包含所有 authorN 字段的检查
            db_conditions.append({"$or": author_or_conditions})
        # 注意：之前的 LIKE "%term%" 逻辑已被移除，替换为精确匹配 $eq

    # Combine all conditions with $and if multiple exist
    if len(db_conditions) > 1:
        final_where = {"$and": db_conditions}
    elif len(db_conditions) == 1:
        final_where = db_conditions[0]
    else:
        final_where = None # No conditions

    print(f"[Debug build_filters] Generated ChromaDB where: {final_where}")
    return final_where

# 系统初始化函数 - 直接连接到特定集合
def initialize_system():
    """系统初始化函数，直接连接到papers0520集合"""
    try:
        start_time = time.time()
        
        # 初始化嵌入模型
        tensor_parallel_size = 1
        print(f"⏳ 初始化嵌入模型 (tensor_parallel_size={tensor_parallel_size})...")
        embedding = VLLMQwenEmbedding(tensor_parallel_size=tensor_parallel_size)
        print("✅ 嵌入模型初始化完成")

        # 设置ChromaDB连接方式
        db_path = "/home/dataset-assist-0/data/chromadb/"
        print(f"💾 连接到数据库: {db_path}")
        chroma_client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # 获取集合
        collection_name = "papers0520"
        print(f"📄 使用集合: {collection_name}")
        collection = chroma_client.get_collection(name=collection_name)
        
        # 创建数据库对象
        database = ChromaDatabase(collection_name=collection_name, dim=embedding.embedding_dim)
        # 替换默认客户端
        database.client = chroma_client
        database.collection = collection
        
        doc_count = collection.count()
        print(f"📊 集合 {collection_name} 中包含 {doc_count} 篇论文")
        
        retriever = SimpleRetriever(embedding, database)
        print("✅ 数据库和检索器初始化完成")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        print(f"✅ 系统初始化完成，用时 {processing_time:.2f} 秒")
        
        return retriever

    except Exception as e:
        print(f"❌ 系统初始化失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# 核心检索函数
def search_papers(query_title, query_abstract, top_k=5, journal=None, min_year=None, max_year=None, author=None):
    """核心检索函数"""
    # Keep the comprehensive debug print statement
    print(f"[Debug Full Inputs] Received: title='{query_title}' ({type(query_title)}), "
          f"abstract='{query_abstract}' ({type(query_abstract)}), "
          f"top_k={top_k} ({type(top_k)}), journal='{journal}' ({type(journal)}), "
          f"min_year='{min_year}' ({type(min_year)}), max_year='{max_year}' ({type(max_year)}), "
          f"author='{author}' ({type(author)})")

    # Restore the original validation and processing logic
    title_present = query_title and query_title.strip()
    abstract_present = query_abstract and query_abstract.strip()
    author_present = author and author.strip()

    if not (title_present or abstract_present or author_present):
        print("[Debug Full Inputs] Validation failed: No non-empty title, abstract, or author provided.")
        return "<div class='output-container'><div style='text-align:center;color:#666;'>⚠️ 请至少输入标题、摘要或作者名称（且不为空）</div></div>"

    # Restore query_text building logic
    query_parts = []
    if title_present:
        query_parts.append(f"Title: {query_title}")
    if abstract_present:
        query_parts.append(f"Abstract: {query_abstract}")

    query_text = "\n".join(query_parts)
    # If only author is present, query_text will be empty.
    # Pass the author name as query_text if title/abstract are empty
    # to satisfy retriever's need for non-empty text for embedding.
    if not query_text and author_present:
        query_text = author
    # Handle the case where all inputs are empty (though validation should prevent this)
    if not query_text:
        # Provide a default or generic query string if absolutely necessary
        # Or rely on the retriever to handle this case if it can.
        # For now, we might let it proceed and potentially fail at the retriever if it requires text.
        # Alternatively, return an error earlier if query_text is mandatory for the retriever.
        print("Warning: query_text is empty after processing inputs. Proceeding...")
        # query_text = "general search" # Example placeholder if needed

    # Restore filter building and retrieval logic
    where_document = build_filters(journal, min_year, max_year, author)
    try:
        print(f"[Debug Full Inputs] Calling retriever.retrieve with query_text='{query_text}', top_k={top_k}, where_document={where_document}")
        # Ensure retriever is accessible (assuming it's initialized globally)
        results = retriever.retrieve(query_text=query_text, top_k=top_k, filter_expression=where_document)
    except Exception as e:
        print(f"❌ Retriever Error: {e}")
        import traceback
        traceback.print_exc()
        return f"<div class='output-container'><div style='text-align:center;color:#666;'>❌ 检索失败: {str(e)}</div></div>"

    if not results:
        return "<div class='output-container'><div style='text-align:center;color:#666;'>🔍 未找到相关论文</div></div>"

    # --- Add Deduplication Logic --- 
    print(f"[Debug] Results before deduplication: {len(results)} items")
    unique_results = []
    seen_titles = set()
    for paper in results:
        entity = paper.get('entity', {})
        title = entity.get('title')
        if title and title not in seen_titles:
            unique_results.append(paper)
            seen_titles.add(title)
    print(f"[Debug] Results after deduplication: {len(unique_results)} items")
    # Use unique_results for display, potentially limiting to top_k again if needed,
    # although the retriever might have already limited based on initial similarity.
    # If we want exactly top_k unique results, we might need to fetch more initially.
    # For now, display all unique results found within the initial fetch.
    display_results = unique_results 
    # --- End Deduplication Logic ---

    # 开始构建 HTML 输出 (use display_results instead of results)
    html_output = ["<div class='output-container'>"]
    for idx, paper in enumerate(display_results, 1):
        entity = paper.get('entity', {})
        summary = entity.get('summary', '')
        link = entity.get('link', '无链接')
        title = entity.get('title', '未知')
        authors_raw = entity.get('authors', '') # Get raw authors string/list
        venue = entity.get('venue', '未知')
        published_year = entity.get('published', '未知')

        # Process authors for clickable links using inline JS
        authors_html = []
        if isinstance(authors_raw, str):
            authors_list = [a.strip() for a in authors_raw.split(',') if a.strip()]
        elif isinstance(authors_raw, list):
            authors_list = authors_raw # Assume it's already a list of strings
        else:
            authors_list = ['未知']

        for author_name in authors_list:
            # Escape quotes for JS string literal and also for HTML attribute value
            js_escaped_author_name = author_name.replace("\\", "\\\\").replace("'", "\\'").replace('"', '&quot;')
            # Escape quotes for the HTML attribute value itself
            html_escaped_js_logic = js_onclick_logic.replace('"', '&quot;').replace("\n", " ") # Replace newlines for HTML attribute
            
            # Construct the onclick attribute with an IIFE embedding the logic
            onclick_attr = f"(function(authorName) {{{html_escaped_js_logic}}})(\'{js_escaped_author_name}\')"
            
            authors_html.append(f'<span class="author-link" onclick="{onclick_attr}">{author_name}</span>')

        authors_display_html = ", ".join(authors_html)

        html_output.append(f"""
            <div class="paper-result">
                <h3>匹配结果 #{idx}</h3>
                <p><strong>📖 标题:</strong> {title}</p>
                <p><strong>📄 摘要:</strong> {summary}</p>
                <p><strong>👥 作者:</strong> {authors_display_html}</p>
                <p><strong>📰 期刊:</strong> {venue}</p>
                <p><strong>📅 年份:</strong> {published_year}</p>
                <p><strong>🔗 链接:</strong> <a href="{link}" target="_blank">{link}</a></p>
            </div>
            """)
    html_output.append("</div>")
    return "".join(html_output)

# 统计函数
def analyze_authors_publications(keyword1, keyword2, keyword3, keyword4, keyword5, keyword6, min_year=None, max_year=None):
    """统计作者在特定领域的论文发表数量"""
    try:
        # 将所有关键词放入列表并清理
        keywords_list = [keyword1, keyword2, keyword3, keyword4, keyword5, keyword6]
        keywords = [kw.strip() for kw in keywords_list if kw.strip()]
        if not keywords:
            return "<div class='output-container'><div style='text-align:center;color:var(--text-color);'>⚠️ 请至少输入一个关键词</div></div>"
            
        # 存储所有检索到的论文，使用字典存储完整论文信息
        papers_dict = {}
        
        # 构建基础过滤条件
        filter_expr = build_filters(min_year=min_year, max_year=max_year)
        # 移除排除特定期刊的条件
        # if filter_expr:
        #     filter_expr += ' AND venue != "nature"'
        # else:
        #     filter_expr = 'venue != "nature"'
            
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
            return "<div class='output-container'><div style='text-align:center;color:var(--text-color);'>🔍 未找到相关论文</div></div>"
        
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
        return f"<div class='output-container'><div style='text-align:center;color:var(--text-color);'>❌ 统计失败: {str(e)}</div></div>"

# CSS样式
css = """
/* 暗色模式检测 */
:root {
    --background-color: #ffffff;
    --text-color: #666666;
    --card-background: #f8f9fa;
    --card-border: #e9ecef;
    --primary-color: #2196f3;
    --primary-hover: #1976d2;
    --title-color: #2c3e50;
    --input-border: #e0e0e0;
    --input-background: #ffffff;
    --container-background: #ffffff;
    --container-shadow: rgba(0,0,0,0.1);
    --group-background: #f8f9fa;
}

@media (prefers-color-scheme: dark) {
    :root {
        --background-color: #1e1e1e;
        --text-color: #cccccc;
        --card-background: #2d2d2d;
        --card-border: #3d3d3d;
        --primary-color: #0e88e8;
        --primary-hover: #0a6eb9;
        --title-color: #e0e0e0;
        --input-border: #555555;
        --input-background: #3d3d3d;
        --container-background: #2d2d2d;
        --container-shadow: rgba(0,0,0,0.3);
        --group-background: #333333;
    }
}

/* 全局样式 */
body, .gradio-container { /* Apply font globally */
    font-family: 'Microsoft YaHei', '微软雅黑', sans-serif !important;
}

.container {
    max-width: 1200px;
    margin: 0 auto;
}

/* 标题样式 */
h1, h2, h3, h4, h5, h6 {
    font-family: "Microsoft YaHei", "微软雅黑", sans-serif !important;
}

h1 {
    text-align: center;
    color: var(--title-color);
    padding: 20px 0;
    margin-bottom: 30px;
    border-bottom: 2px solid var(--card-border);
}

/* 输入面板样式 */
.input-container {
    background: var(--container-background);
    border-radius: 10px;
    padding: 20px;
    box-shadow: 0 2px 6px var(--container-shadow);
    margin-bottom: 20px;
}

/* 输入框样式 */
.gr-textbox, .gr-textarea {
    border: 1px solid var(--input-border) !important;
    border-radius: 8px !important;
    padding: 10px !important;
    font-size: 14px !important;
    transition: all 0.3s ease;
    background-color: var(--input-background) !important;
    color: var(--text-color) !important;
}

.gr-textbox:focus, .gr-textarea:focus {
    border-color: var(--primary-color) !important;
    box-shadow: 0 0 0 2px rgba(33,150,243,0.1) !important;
}

/* 按钮样式 */
.gr-button {
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    font-family: "Microsoft YaHei", "微软雅黑", sans-serif !important;
}

.gr-button.primary {
    background: var(--primary-color) !important;
    border: none !important;
}

.gr-button.primary:hover {
    background: var(--primary-hover) !important;
    transform: translateY(-1px);
}

/* 结果面板样式 */
.output-container {
    background: var(--container-background);
    border-radius: 10px;
    padding: 20px !important;
    height: 800px !important;
    overflow-y: auto;
    box-shadow: 0 2px 6px var(--container-shadow);
    color: var(--text-color);
}

.paper-result {
    background: var(--card-background);
    border-radius: 8px;
    padding: 15px !important;
    margin: 15px 0 !important;
    border: 1px solid var(--card-border);
    transition: all 0.3s ease;
}

.paper-result:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px var(--container-shadow);
}

.paper-result h3 {
    color: var(--primary-color);
    margin: 0 0 15px 0 !important;
    font-size: 18px !important;
}

.paper-result p {
    margin: 8px 0 !important;
    line-height: 1.5;
}

.paper-result a {
    color: var(--primary-color);
}

/* 分组样式 */
.search-group {
    background: var(--group-background);
    border-radius: 8px;
    padding: 15px;
    margin-bottom: 15px;
}

.search-group-title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 15px;
    color: var(--title-color);
}

/* 统计结果样式 */
.stats-container {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
    gap: 15px;
    padding: 15px;
}

.author-stat-card {
    background: var(--container-background);
    border-radius: 8px;
    padding: 15px;
    box-shadow: 0 2px 4px var(--container-shadow);
    transition: all 0.3s ease;
}

.author-stat-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.author-stat-card h3 {
    color: var(--primary-color);
    margin: 0 0 10px 0;
    font-size: 16px;
}

.paper-count {
    font-size: 14px;
    color: var(--text-color);
    margin: 0;
}

/* 标签页样式 */
.tab-selected {
    background: var(--primary-color) !important;
    color: white !important;
}

/* Style for clickable authors */
.author-link {
    color: var(--primary-color); /* Use primary color for link */
    text-decoration: underline;
    cursor: pointer;
    margin-right: 5px; /* Add some spacing between names */
}
.author-link:hover {
    color: var(--link-hover-color); /* Define a hover color if needed */
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
                        # Restore title and abstract inputs
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
                        
                        with gr.Accordion("🔍 高级筛选", open=True):
                            with gr.Column(elem_classes="search-group"):
                                # Ensure author_input is defined here (or moved from base search if duplicated)
                                author_input = gr.Textbox(
                                    label="作者姓名",
                                    placeholder="输入完整的作者姓名（精确匹配）...",
                                )
                                journal_input = gr.Textbox(
                                    label="目标期刊",
                                    placeholder="输入期刊名称（选填）...",
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
                        
                        with gr.Row():
                            top_k_input = gr.Slider(
                                minimum=1,
                                maximum=30,
                                value=5,
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
                        value="<div class='output-container'><div style='text-align:center;color:var(--text-color);padding:20px;'>等待检索，请输入搜索条件...</div></div>"
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
                    value="<div class='output-container'><div style='text-align:center;color:var(--text-color);'>等待统计...</div></div>"
                )
    
    # 事件绑定
    search_btn.click(
        fn=search_papers,
        inputs=[
            title_input,      # 1. query_title
            abstract_input,   # 2. query_abstract
            top_k_input,      # 3. top_k
            journal_input,    # 4. journal
            min_year_input,   # 5. min_year
            max_year_input,   # 6. max_year
            author_input      # 7. author
        ],
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
    parser.add_argument('--port', type=int, default=8081, help='服务端口号')
    parser.add_argument('--no-share', action='store_true', help='不创建公共链接')
    args = parser.parse_args()
    
    # 系统初始化
    print(f"🚀 系统启动 - 端口: {args.port}")
    retriever = initialize_system()
    
    # 启动界面
    interface.launch(server_port=args.port, share=not args.no_share)