import chromadb
from chromadb.config import Settings
import os
import time

def check_database():
    """检查ChromaDB数据库中的所有集合和数据"""
    # 要检查的数据库路径列表
    db_paths = [
        "/home/dataset-assist-0/data/chromadb",
        "/home/dataset-assist-0/data/chromadb/7782422d-e220-45cb-8df6-5843362d438f"
    ]
    
    for db_path in db_paths:
        if not os.path.exists(db_path):
            print(f"\n⚠️ 路径不存在: {db_path}")
            continue
            
        print(f"\n{'=' * 80}")
        print(f"📂 检查数据库: {db_path}")
        print(f"{'=' * 80}")
        
        # 连接到数据库
        client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(
                anonymized_telemetry=False
            )
        )
        
        # 获取所有集合
        collections = client.list_collections()
        print(f"\n📚 发现 {len(collections)} 个集合")
        
        # 检查每个集合
        for collection in collections:
            col_name = collection.name
            col_id = collection.id
            
            # 获取集合元数据
            try:
                metadata = collection.metadata
                dimension = metadata.get("dimension", "未知") if metadata else "未知"
            except:
                dimension = "未知"
                
            # 获取文档数量
            count = collection.count()
            
            print(f"\n📑 集合: {col_name} (ID: {col_id})")
            print(f"📊 文档数量: {count}")
            print(f"📏 向量维度: {dimension}")
            
            # 如果没有数据，跳过后续步骤
            if count == 0:
                print("❌ 集合为空")
                continue
                
            # 获取样本数据
            start_time = time.time()
            print("⏳ 正在获取样本数据...")
            
            # 获取前5条和后5条数据的ID
            try:
                # 获取所有ID
                all_ids = collection.get(include=[])['ids']
                first_ids = all_ids[:5]
                last_ids = all_ids[-5:] if count > 5 else []
                
                print(f"🔍 前5个ID: {first_ids}")
                if last_ids and last_ids != first_ids:
                    print(f"🔍 后5个ID: {last_ids}")
                    
                # 获取ID模式 (检查是否使用了自增ID)
                if all(id.startswith("doc_") for id in first_ids):
                    print("✅ 使用自增ID模式")
                else:
                    print("⚠️ 未使用自增ID模式")
                
                # 获取一条示例数据
                sample = collection.get(ids=[first_ids[0]], include=["metadatas"])
                if sample and 'metadatas' in sample and sample['metadatas']:
                    metadata = sample['metadatas'][0]
                    print("\n📝 示例文档:")
                    for key, value in metadata.items():
                        value_str = str(value)
                        if len(value_str) > 100:
                            value_str = value_str[:100] + "..."
                        print(f"  {key}: {value_str}")
            except Exception as e:
                print(f"❌ 获取样本数据失败: {str(e)}")
            
            elapsed = time.time() - start_time
            print(f"⏱️ 获取样本耗时: {elapsed:.2f}秒")
            
            # 获取数据分布
            if count > 10:
                # 按年份统计 (如果有published字段)
                try:
                    # 随机获取100条数据样本
                    sample_size = min(100, count)
                    sample_ids = all_ids[:sample_size]
                    samples = collection.get(ids=sample_ids, include=["metadatas"])
                    
                    # 按年份统计
                    years = {}
                    venues = {}
                    
                    for metadata in samples['metadatas']:
                        if 'published' in metadata:
                            year = metadata['published']
                            years[year] = years.get(year, 0) + 1
                        
                        if 'venue' in metadata:
                            venue = metadata['venue']
                            venues[venue] = venues.get(venue, 0) + 1
                    
                    if years:
                        print("\n📅 年份分布 (基于样本):")
                        for year, count in sorted(years.items(), reverse=True)[:5]:
                            print(f"  {year}: {count}篇")
                    
                    if venues:
                        print("\n🏛️ 期刊分布 (基于样本):")
                        for venue, count in sorted(venues.items(), key=lambda x: x[1], reverse=True)[:5]:
                            print(f"  {venue}: {count}篇")
                except Exception as e:
                    print(f"❌ 获取数据分布失败: {str(e)}")
    
    print("\n✅ 数据库检查完成")

if __name__ == "__main__":
    check_database() 