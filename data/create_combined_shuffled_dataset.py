#!/usr/bin/env python3
"""
合并和shuffle数据集的脚本
将forml4和mma数据集合并后进行shuffle，确保训练时数据分布均匀
"""

import json
import random
import os
from pathlib import Path

def read_jsonl(file_path):
    """读取JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def write_jsonl(data, file_path):
    """写入JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def main():
    # 定义文件路径
    forml4_path = "forml4/train.jsonl"
    mma_path = "mma/lean_train.jsonl"
    output_dir = "mma_forml4"
    output_path = os.path.join(output_dir, "train.jsonl")
    
    print("=" * 60)
    print("合并和Shuffle数据集")
    print("=" * 60)
    
    # 检查输入文件是否存在
    if not os.path.exists(forml4_path):
        print(f"❌ 错误: 找不到文件 {forml4_path}")
        return
    
    if not os.path.exists(mma_path):
        print(f"❌ 错误: 找不到文件 {mma_path}")
        return
    
    # 读取数据集
    print(f"📖 正在读取 {forml4_path}...")
    forml4_data = read_jsonl(forml4_path)
    print(f"   - forml4数据集: {len(forml4_data):,} 条")
    
    print(f"📖 正在读取 {mma_path}...")
    mma_data = read_jsonl(mma_path)
    print(f"   - mma数据集: {len(mma_data):,} 条")
    
    # 合并数据
    print("🔗 正在合并数据集...")
    combined_data = forml4_data + mma_data
    print(f"   - 合并后总计: {len(combined_data):,} 条")
    
    # 设置随机种子并shuffle
    print("🔀 正在shuffle数据集...")
    random.seed(42)  # 设置随机种子确保可复现
    random.shuffle(combined_data)
    print("   - Shuffle完成")
    
    # 验证数据格式
    print("🔍 验证数据格式...")
    sample = combined_data[0]
    required_keys = ['input', 'output']
    for key in required_keys:
        if key not in sample:
            print(f"❌ 错误: 数据缺少必要字段 '{key}'")
            return
    print("   - 数据格式验证通过")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存合并和shuffle后的数据
    print(f"💾 正在保存到 {output_path}...")
    write_jsonl(combined_data, output_path)
    
    # 验证保存的文件
    print("✅ 验证保存的文件...")
    with open(output_path, 'r') as f:
        lines = f.readlines()
    saved_count = len([line for line in lines if line.strip()])
    print(f"   - 保存的数据条数: {saved_count:,}")
    
    if saved_count == len(combined_data):
        print("✅ 文件保存成功!")
    else:
        print("❌ 文件保存可能有问题，数据条数不匹配")
        return
    
    # 显示前几条和后几条数据的来源分布
    print("\n📊 数据分布检查:")
    print("前10条数据来源:")
    for i in range(min(10, len(combined_data))):
        # 简单判断：如果input长度较短且包含特定关键词，可能是forml4
        input_text = combined_data[i]['input']
        if len(input_text) < 200 and 'association' in input_text:
            source = "forml4"
        elif 'Statement in natural language' in input_text:
            source = "forml4"
        else:
            source = "mma"
        print(f"   {i+1:2d}. {source} - {input_text[:50]}...")
    
    print("\n后10条数据来源:")
    for i in range(max(0, len(combined_data)-10), len(combined_data)):
        input_text = combined_data[i]['input']
        if len(input_text) < 200 and 'association' in input_text:
            source = "forml4"
        elif 'Statement in natural language' in input_text:
            source = "forml4"
        else:
            source = "mma"
        print(f"   {i+1:2d}. {source} - {input_text[:50]}...")
    
    print("\n" + "=" * 60)
    print("🎉 数据集合并和shuffle完成!")
    print(f"📁 输出文件: {output_path}")
    print(f"📈 总数据量: {len(combined_data):,} 条")
    print(f"🔀 已使用随机种子42进行shuffle")
    print("=" * 60)

if __name__ == "__main__":
    main() 