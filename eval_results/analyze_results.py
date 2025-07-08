#!/usr/bin/env python3
import json
import numpy as np
from collections import defaultdict

def analyze_results(file_path):
    """Analyze the evaluation results and print metrics."""
    
    # Load results
    results = []
    with open(file_path, 'r') as f:
        for line in f:
            results.append(json.loads(line.strip()))
    
    print(f"Total samples: {len(results)}")
    
    # Extract scores and labels
    alignment_scores = [r['alignment_score'] for r in results]
    certainty_scores = [r['certainty_score'] for r in results]
    similarity_scores = [r['similarity_score'] for r in results]
    labels = [r['label'] for r in results]
    
    # Convert to numpy arrays
    alignment_scores = np.array(alignment_scores)
    certainty_scores = np.array(certainty_scores)
    similarity_scores = np.array(similarity_scores)
    labels = np.array(labels)
    
    # Basic statistics
    print(f"\nScore Statistics:")
    print(f"Alignment Score - Mean: {alignment_scores.mean():.4f}, Std: {alignment_scores.std():.4f}")
    print(f"Certainty Score - Mean: {certainty_scores.mean():.4f}, Std: {certainty_scores.std():.4f}")
    print(f"Similarity Score - Mean: {similarity_scores.mean():.4f}, Std: {similarity_scores.std():.4f}")
    
    # Class distribution
    pos_mask = labels == True
    neg_mask = labels == False
    print(f"\nData Balance:")
    print(f"Positive samples: {pos_mask.sum()} ({pos_mask.sum()/len(labels):.1%})")
    print(f"Negative samples: {neg_mask.sum()} ({neg_mask.sum()/len(labels):.1%})")
    
    # Score analysis by class
    if pos_mask.sum() > 0:
        print(f"\nPositive samples (label=True):")
        print(f"  Alignment Score - Mean: {alignment_scores[pos_mask].mean():.4f}, Std: {alignment_scores[pos_mask].std():.4f}")
        print(f"  Certainty Score - Mean: {certainty_scores[pos_mask].mean():.4f}, Std: {certainty_scores[pos_mask].std():.4f}")
        print(f"  Similarity Score - Mean: {similarity_scores[pos_mask].mean():.4f}, Std: {similarity_scores[pos_mask].std():.4f}")
    
    if neg_mask.sum() > 0:
        print(f"\nNegative samples (label=False):")
        print(f"  Alignment Score - Mean: {alignment_scores[neg_mask].mean():.4f}, Std: {alignment_scores[neg_mask].std():.4f}")
        print(f"  Certainty Score - Mean: {certainty_scores[neg_mask].mean():.4f}, Std: {certainty_scores[neg_mask].std():.4f}")
        print(f"  Similarity Score - Mean: {similarity_scores[neg_mask].mean():.4f}, Std: {similarity_scores[neg_mask].std():.4f}")
    
    # Calculate metrics for different thresholds
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    print(f"\nAlignment Score Metrics:")
    print("Threshold | Accuracy | Precision | Recall | F1")
    print("-" * 50)
    
    best_f1 = 0.0
    best_threshold = 0.5
    
    for threshold in thresholds:
        predictions = alignment_scores > threshold
        
        # Calculate metrics
        tp = np.sum((predictions == True) & (labels == True))
        fp = np.sum((predictions == True) & (labels == False))
        fn = np.sum((predictions == False) & (labels == True))
        tn = np.sum((predictions == False) & (labels == False))
        
        accuracy = (tp + tn) / len(labels)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
        
        print(f"{threshold:8.1f} | {accuracy:8.4f} | {precision:9.4f} | {recall:6.4f} | {f1:6.4f}")
    
    print(f"\nBest threshold: {best_threshold} (F1: {best_f1:.4f})")
    
    # Score separation analysis
    if pos_mask.sum() > 0 and neg_mask.sum() > 0:
        pos_mean = alignment_scores[pos_mask].mean()
        neg_mean = alignment_scores[neg_mask].mean()
        score_separation = pos_mean - neg_mean
        
        print(f"\nScore Separation Analysis:")
        print(f"Positive samples score mean: {pos_mean:.4f}")
        print(f"Negative samples score mean: {neg_mean:.4f}")
        print(f"Score separation (pos - neg): {score_separation:.4f}")
        
        if score_separation > 0:
            print("✓ Model correctly assigns higher scores to positive samples")
        else:
            print("✗ Model assigns higher scores to negative samples (potential issue)")

if __name__ == "__main__":
    file_path = "alisa/FormalAlign/eval_results/alignment/alignment_scores_minif2f_test_mma_forml4_combined.jsonl"
    analyze_results(file_path) 