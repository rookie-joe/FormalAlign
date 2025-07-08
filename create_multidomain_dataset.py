#!/usr/bin/env python3
"""
Create multi-domain dataset by combining MMA+FormL4 and MiniF2F data
to improve model generalization across different mathematical domains.
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any

def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def save_jsonl(data: List[Dict[str, Any]], file_path: str):
    """Save data to JSONL file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def create_multidomain_dataset(
    mma_forml4_path: str,
    minif2f_path: str,
    output_dir: str,
    mma_forml4_ratio: float = 0.7,
    minif2f_ratio: float = 0.3,
    seed: int = 42
):
    """
    Create multi-domain dataset by combining MMA+FormL4 and MiniF2F data.
    
    Args:
        mma_forml4_path: Path to MMA+FormL4 training data
        minif2f_path: Path to MiniF2F training data  
        output_dir: Output directory for combined dataset
        mma_forml4_ratio: Ratio of MMA+FormL4 data in final dataset
        minif2f_ratio: Ratio of MiniF2F data in final dataset
        seed: Random seed for reproducibility
    """
    random.seed(seed)
    
    # Load data from both domains
    print(f"Loading MMA+FormL4 data from {mma_forml4_path}")
    mma_forml4_data = load_jsonl(mma_forml4_path)
    print(f"Loaded {len(mma_forml4_data)} MMA+FormL4 samples")
    
    print(f"Loading MiniF2F data from {minif2f_path}")
    minif2f_data = load_jsonl(minif2f_path)
    print(f"Loaded {len(minif2f_data)} MiniF2F samples")
    
    # Sample data according to ratios
    total_samples = len(mma_forml4_data) + len(minif2f_data)
    target_mma_forml4 = int(total_samples * mma_forml4_ratio)
    target_minif2f = int(total_samples * minif2f_ratio)
    
    # Sample data (with replacement if needed)
    sampled_mma_forml4 = random.sample(
        mma_forml4_data, 
        min(target_mma_forml4, len(mma_forml4_data))
    )
    
    sampled_minif2f = random.sample(
        minif2f_data,
        min(target_minif2f, len(minif2f_data))
    )
    
    # Combine datasets
    combined_data = sampled_mma_forml4 + sampled_minif2f
    
    # Shuffle the combined dataset
    random.shuffle(combined_data)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save combined dataset
    train_path = output_path / "train.jsonl"
    save_jsonl(combined_data, str(train_path))
    
    # Create validation set (10% of combined data)
    val_size = len(combined_data) // 10
    val_data = combined_data[:val_size]
    train_data = combined_data[val_size:]
    
    val_path = output_path / "val.jsonl"
    save_jsonl(val_data, str(val_path))
    
    # Save final training data
    save_jsonl(train_data, str(train_path))
    
    print(f"Created multi-domain dataset:")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - Validation samples: {len(val_data)}")
    print(f"  - MMA+FormL4 samples: {len(sampled_mma_forml4)}")
    print(f"  - MiniF2F samples: {len(sampled_minif2f)}")
    print(f"  - Output directory: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create multi-domain dataset")
    parser.add_argument("--mma_forml4_path", type=str, 
                       default="../data/mma_forml4/train.jsonl",
                       help="Path to MMA+FormL4 training data")
    parser.add_argument("--minif2f_path", type=str,
                       default="../data/minif2f/train.jsonl", 
                       help="Path to MiniF2F training data")
    parser.add_argument("--output_dir", type=str,
                       default="../data/multidomain_combined",
                       help="Output directory for combined dataset")
    parser.add_argument("--mma_forml4_ratio", type=float, default=0.7,
                       help="Ratio of MMA+FormL4 data in final dataset")
    parser.add_argument("--minif2f_ratio", type=float, default=0.3,
                       help="Ratio of MiniF2F data in final dataset")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    create_multidomain_dataset(
        args.mma_forml4_path,
        args.minif2f_path,
        args.output_dir,
        args.mma_forml4_ratio,
        args.minif2f_ratio,
        args.seed
    )

if __name__ == "__main__":
    main() 