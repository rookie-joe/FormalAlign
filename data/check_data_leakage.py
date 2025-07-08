#!/usr/bin/env python3
"""
检查训练数据中是否包含测试数据的数据泄露检测脚本
优化版本：只检查input的完全匹配，加速搜索，并移除重复数据

python check_data_leakage.py \
    --train_file mma_forml4/train.jsonl \
    --test_files forml4/misalignment/formatted_random_test_clip.jsonl \
    --output_dir clean_set

python check_data_leakage.py \
    --train_file mma_forml4/train.jsonl \
    --test_files forml4/misalignment/formatted_basic_test_clip.jsonl \
    --output_dir clean_set

python check_data_leakage.py \
    --train_file mma_forml4/train.jsonl \
    --test_files minif2f/misalignment/valid_clip.jsonl \
    --output_dir clean_set

ALL：

python check_data_leakage.py \
    --train_file mma_forml4/train.jsonl \
    --test_files forml4/misalignment/formatted_random_test_clip.jsonl forml4/misalignment/formatted_basic_test_clip.jsonl minif2f/misalignment/valid_clip.jsonl minif2f/misalignment/test_clip.jsonl \
    --output_dir clean_set
"""

import json
import hashlib
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple
import argparse
import re

def normalize_text(text: str) -> str:
    """
    标准化文本，去除空格、换行符等，便于比较
    """
    # 去除多余空白字符
    text = re.sub(r'\s+', ' ', text.strip())
    # 转换为小写
    text = text.lower()
    return text

def calculate_hash(text: str) -> str:
    """
    计算文本的哈希值
    """
    return hashlib.md5(text.encode('utf-8')).hexdigest()

def read_jsonl(file_path: str) -> List[Dict]:
    """
    读取JSONL文件
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: JSON decode error at line {line_num}: {e}")
    return data

def write_jsonl(data: List[Dict], file_path: str):
    """
    写入JSONL文件
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def extract_input_from_training_data(item: Dict) -> str:
    """
    从训练数据中提取input文本
    """
    if 'input' in item:
        return normalize_text(item['input'])
    else:
        return normalize_text(json.dumps(item, sort_keys=True))

def extract_input_from_test_data(item: Dict) -> str:
    """
    从测试数据中提取input文本
    """
    if 'input' in item:
        return normalize_text(item['input'])
    else:
        return normalize_text(json.dumps(item, sort_keys=True))

def find_exact_matches(train_data: List[Dict], test_data: List[Dict]) -> Tuple[List[Tuple[int, int, str]], Set[int]]:
    """
    查找input完全匹配的数据，返回匹配列表和重复的测试数据索引
    """
    matches = []
    duplicate_test_indices = set()
    
    # 为训练数据创建input哈希映射
    train_hashes = {}
    for i, item in enumerate(train_data):
        input_text = extract_input_from_training_data(item)
        input_hash = calculate_hash(input_text)
        train_hashes[input_hash] = i
    
    print(f"   - 训练数据input哈希数量: {len(train_hashes):,}")
    
    # 检查测试数据
    for j, test_item in enumerate(test_data):
        input_text = extract_input_from_test_data(test_item)
        input_hash = calculate_hash(input_text)
        
        if input_hash in train_hashes:
            train_idx = train_hashes[input_hash]
            matches.append((train_idx, j, input_text[:100] + "..."))
            duplicate_test_indices.add(j)
    
    return matches, duplicate_test_indices

def create_clean_dataset(test_data: List[Dict], duplicate_indices: Set[int], output_file: str) -> int:
    """
    创建清理后的数据集，移除重复的数据
    """
    clean_data = []
    removed_count = 0
    
    for i, item in enumerate(test_data):
        if i not in duplicate_indices:
            clean_data.append(item)
        else:
            removed_count += 1
    
    # 写入清理后的数据
    write_jsonl(clean_data, output_file)
    
    return removed_count

def analyze_data_distribution(train_file: str, test_files: List[str], output_dir: str) -> Dict:
    """
    分析数据分布并创建清理后的数据集
    """
    print("=" * 60)
    print("数据泄露分析 (仅检查input完全匹配)")
    print("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取训练数据
    print(f"📖 读取训练数据: {train_file}")
    train_data = read_jsonl(train_file)
    print(f"   - 训练数据条数: {len(train_data):,}")
    
    # 分析训练数据格式
    if train_data:
        sample = train_data[0]
        print(f"   - 训练数据格式: {list(sample.keys())}")
    
    results = {
        'train_file': train_file,
        'train_count': len(train_data),
        'test_files': {},
        'exact_matches': {},
        'cleaned_files': {}
    }
    
    # 检查每个测试文件
    for test_file in test_files:
        print(f"\n📖 读取测试数据: {test_file}")
        test_data = read_jsonl(test_file)
        print(f"   - 测试数据条数: {len(test_data):,}")
        
        if test_data:
            sample = test_data[0]
            print(f"   - 测试数据格式: {list(sample.keys())}")
        
        # 查找完全匹配
        print(f"🔍 查找input完全匹配...")
        exact_matches, duplicate_indices = find_exact_matches(train_data, test_data)
        results['exact_matches'][test_file] = exact_matches
        print(f"   - 完全匹配数量: {len(exact_matches)}")
        
        if exact_matches:
            print("   - 完全匹配详情:")
            for train_idx, test_idx, text in exact_matches[:10]:  # 显示前10个
                print(f"     Train[{train_idx}] <-> Test[{test_idx}]: {text}")
            if len(exact_matches) > 10:
                print(f"     ... 还有 {len(exact_matches) - 10} 个匹配")
        
        # 创建清理后的数据集
        print(f"🧹 创建清理后的数据集...")
        test_filename = os.path.basename(test_file)
        clean_filename = f"clean_{test_filename}"
        clean_file_path = os.path.join(output_dir, clean_filename)
        
        removed_count = create_clean_dataset(test_data, duplicate_indices, clean_file_path)
        clean_count = len(test_data) - removed_count
        
        print(f"   - 原始数据: {len(test_data):,} 条")
        print(f"   - 移除重复: {removed_count:,} 条")
        print(f"   - 清理后数据: {clean_count:,} 条")
        print(f"   - 清理后文件: {clean_file_path}")
        
        results['test_files'][test_file] = {
            'count': len(test_data),
            'format': list(sample.keys()) if test_data else [],
            'removed_count': removed_count,
            'clean_count': clean_count
        }
        
        results['cleaned_files'][test_file] = {
            'clean_file': clean_file_path,
            'removed_count': removed_count,
            'clean_count': clean_count
        }
    
    return results

def save_results(results: Dict, output_file: str):
    """
    保存分析结果
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 分析结果已保存到: {output_file}")

def print_summary(results: Dict):
    """
    打印分析总结
    """
    print("\n" + "=" * 60)
    print("📊 数据泄露分析总结")
    print("=" * 60)
    
    train_file = results['train_file']
    train_count = results['train_count']
    
    print(f"训练数据: {train_file}")
    print(f"训练数据总量: {train_count:,} 条")
    
    total_exact_matches = 0
    total_removed = 0
    total_clean = 0
    
    for test_file, test_info in results['test_files'].items():
        test_count = test_info['count']
        exact_count = len(results['exact_matches'][test_file])
        removed_count = test_info['removed_count']
        clean_count = test_info['clean_count']
        
        total_exact_matches += exact_count
        total_removed += removed_count
        total_clean += clean_count
        
        print(f"\n测试数据: {test_file}")
        print(f"  测试数据总量: {test_count:,} 条")
        print(f"  完全匹配: {exact_count} 条 ({exact_count/test_count*100:.2f}%)")
        print(f"  移除重复: {removed_count:,} 条")
        print(f"  清理后数据: {clean_count:,} 条")
        
        if exact_count > 0:
            print(f"  🚨 发现数据泄露！已移除重复数据")
        else:
            print(f"  ✅ 未发现数据泄露")
    
    print(f"\n总体统计:")
    print(f"  总完全匹配: {total_exact_matches} 条")
    print(f"  总移除数据: {total_removed:,} 条")
    print(f"  总清理后数据: {total_clean:,} 条")
    
    if total_exact_matches > 0:
        print(f"  🚨 发现 {total_exact_matches} 条完全匹配的数据泄露！")
        print(f"  ✅ 已成功移除重复数据，清理后的数据集已保存")
    else:
        print(f"  ✅ 未发现明显的数据泄露问题")

def main():
    parser = argparse.ArgumentParser(description="检查训练数据中是否包含测试数据并创建清理后的数据集")
    parser.add_argument("--train_file", type=str, required=True, help="训练数据文件路径")
    parser.add_argument("--test_files", type=str, nargs="+", required=True, help="测试数据文件路径列表")
    parser.add_argument("--output_dir", type=str, default="clean_set", help="清理后数据集的输出目录")
    parser.add_argument("--results_file", type=str, default="clean_set/data_leakage_analysis.json", help="分析结果文件")
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.train_file):
        print(f"❌ 错误: 训练文件不存在: {args.train_file}")
        return
    
    for test_file in args.test_files:
        if not os.path.exists(test_file):
            print(f"❌ 错误: 测试文件不存在: {test_file}")
            return
    
    # 执行分析
    results = analyze_data_distribution(args.train_file, args.test_files, args.output_dir)
    
    # 保存结果
    save_results(results, args.results_file)
    
    # 打印总结
    print_summary(results)
    
    print(f"\n🎉 清理完成！")
    print(f"📁 清理后的数据集保存在: {args.output_dir}/")
    print(f"📊 详细分析结果保存在: {args.results_file}")

if __name__ == "__main__":
    main() 