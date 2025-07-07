import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from tqdm import tqdm
from accelerate import Accelerator
import os
import numpy as np
import torch.nn.functional as F
from utils.verifier_models import load_verifier_clip
from utils.datasets import make_test_verifierclip_data_module, make_testing_dataloader

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    fp16: Optional[bool] = field(default=False)
    project_dim: Optional[int] = field(default=512)

@dataclass
class DataArguments:
    data_dir: str = field(default='data/gsm8k/model_generation')
    target_set: str = field(default='test')
    generator_id: str = field(default='llama7b-2-ep2')
    data_id: str = field(default='test')
    verifier_id: str = field(default='default')
    output_dir: str = field(default='eval_results/alignment')

@dataclass
class InferenceArguments:
    batch_size: int = field(default=1)
    seed: int = field(default=None)
    per_device_eval_batch_size: int = field(default=4)
    use_autoregressive_certainty: bool = field(default=True, metadata={"help": "Whether to use autoregressive certainty score calculation (recommended) or teacher forcing"})
    max_new_tokens: int = field(default=512, metadata={"help": "Maximum number of tokens to generate for autoregressive certainty calculation"})

def calculate_certainty_score_autoregressive(model, tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, t_eoss: torch.Tensor, max_new_tokens: int = 512) -> torch.Tensor:
    """
    Calculate certainty score using autoregressive generation:
    V_cer = exp(1/n * sum(log P(FL_i,j | FL_i,<j, NL_i)))
    
    This version feeds only [NL] and generates FL autoregressively,
    calculating the sequence-level log probability.
    """
    device = input_ids.device
    batch_size = input_ids.size(0)
    certainty_scores = []
    
    for i in range(batch_size):
        # Get the NL part (up to t_eoss)
        t_eos_val = t_eoss[i].item() if hasattr(t_eoss[i], 'item') else t_eoss[i]
        nl_input_ids = input_ids[i:i+1, :t_eos_val]  # [1, nl_len]
        nl_attention_mask = attention_mask[i:i+1, :t_eos_val]  # [1, nl_len]
        
        # Get the ground truth FL part for comparison
        ground_truth_fl = input_ids[i, t_eos_val:]  # [fl_len]
        
        # Remove padding tokens from ground truth
        # Find the first padding token (assuming tokenizer.pad_token_id exists)
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        # Find actual FL length (excluding padding)
        fl_length = len(ground_truth_fl)
        for j, token in enumerate(ground_truth_fl):
            if token == pad_token_id:
                fl_length = j
                break
        
        if fl_length == 0:
            # No FL tokens, assign very low certainty
            certainty_scores.append(torch.tensor(0.001, device=device))
            continue
            
        ground_truth_fl = ground_truth_fl[:fl_length]  # Remove padding
        
        # Generate FL autoregressively and calculate log probabilities
        current_input = nl_input_ids.clone()  # [1, nl_len]
        current_attention_mask = nl_attention_mask.clone()  # [1, nl_len]
        
        log_probs = []
        
        for step in range(min(fl_length, max_new_tokens)):
            # Get model predictions for next token
            with torch.no_grad():
                outputs = model.backbone(
                    input_ids=current_input,
                    attention_mask=current_attention_mask,
                    return_dict=True
                )
                
                # Get logits for the last position
                next_token_logits = outputs.logits[0, -1, :]  # [vocab_size]
                
                # Get log probabilities
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Get ground truth next token
                if step < len(ground_truth_fl):
                    ground_truth_next_token = ground_truth_fl[step]
                    
                    # Record log probability of ground truth token
                    token_log_prob = next_token_log_probs[ground_truth_next_token]
                    log_probs.append(token_log_prob)
                    
                    # Append ground truth token to input for next step
                    current_input = torch.cat([
                        current_input,
                        ground_truth_next_token.unsqueeze(0).unsqueeze(0)
                    ], dim=1)
                    
                    # Extend attention mask
                    current_attention_mask = torch.cat([
                        current_attention_mask,
                        torch.ones(1, 1, device=device, dtype=current_attention_mask.dtype)
                    ], dim=1)
                else:
                    break
        
        if len(log_probs) == 0:
            # No tokens generated, assign very low certainty
            certainty_scores.append(torch.tensor(0.001, device=device))
        else:
            # Calculate average log probability and take exponential
            avg_log_prob = torch.stack(log_probs).mean()
            certainty_score = torch.exp(avg_log_prob)
            certainty_scores.append(certainty_score)
    
    return torch.stack(certainty_scores)

def calculate_certainty_score(logits: torch.Tensor, input_ids: torch.Tensor, attention_mask: torch.Tensor, t_eoss: torch.Tensor) -> torch.Tensor:
    """
    Calculate certainty score based on equation (1) in the paper:
    V_cer = exp(1/n * sum(log P(FL_i,j | FL_i,<j, NL_i)))
    Only calculate for FL tokens (after t_eoss position)
    
    NOTE: This is the old teacher forcing implementation.
    Use calculate_certainty_score_autoregressive for better evaluation.
    """
    # Get log probabilities for next token prediction
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)  # [batch_size, seq_len-1, vocab_size]
    
    # Get the actual next tokens
    next_tokens = input_ids[:, 1:]  # [batch_size, seq_len-1]
    
    # Get the log probs of the actual next tokens
    token_log_probs = torch.gather(log_probs, -1, next_tokens.unsqueeze(-1)).squeeze(-1)  # [batch_size, seq_len-1]
    
    # Create a mask for the formal part (after t_eoss)
    batch_size = input_ids.size(0)
    seq_length = token_log_probs.size(1)
    formal_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
    
    for i in range(batch_size):
        # t_eoss[i] is the position where formal language starts
        # We want to mask from t_eoss[i] to the end (excluding padding)
        t_eos_val = t_eoss[i].item() if hasattr(t_eoss[i], 'item') else t_eoss[i]
        # Start from t_eoss position (since we're predicting next token)
        if t_eos_val < seq_length:
            formal_mask[i, t_eos_val:] = True
    
    # Apply both attention mask and formal mask
    # attention_mask[:, 1:] corresponds to positions where we have valid next tokens
    valid_mask = attention_mask[:, 1:] * formal_mask
    
    # Calculate average log prob per sequence (only for the formal part)
    seq_lengths = valid_mask.sum(dim=1)
    masked_log_probs = token_log_probs * valid_mask
    
    # Avoid division by zero - if no FL tokens, return very low certainty
    avg_log_probs = torch.where(
        seq_lengths > 0,
        masked_log_probs.sum(dim=1) / seq_lengths.clamp(min=1),
        torch.tensor(-10.0, device=token_log_probs.device)  # Low certainty for empty FL
    )
    
    # Return exponential of average log probs
    return torch.exp(avg_log_probs)

def calculate_similarity_score(nl_hidden: torch.Tensor, fl_hidden: torch.Tensor) -> torch.Tensor:
    """
    Calculate similarity score based on equation (2) in the paper:
    V_sim = cos(Z_φ(NL_i), Z_φ(FL_i | NL_i))
    Note: Assumes input embeddings are already normalized
    """
    # Calculate cosine similarity (inputs should already be normalized)
    return torch.sum(nl_hidden * fl_hidden, dim=-1)

def calculate_alignment_score(certainty_score: torch.Tensor, similarity_score: torch.Tensor) -> torch.Tensor:
    """
    Calculate overall alignment score based on equation (3) in the paper:
    V_align = (V_cer + V_sim) / 2
    """
    return (certainty_score + similarity_score) / 2

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InferenceArguments))
    model_args, data_args, inference_args = parser.parse_args_into_dataclasses()

    # Initialize accelerator
    accelerator = Accelerator()

    # Load model and tokenizer
    model, tokenizer = load_verifier_clip(model_args)
    model.eval()

    # Prepare dataset and dataloader
    dataset = make_test_verifierclip_data_module(tokenizer, data_args)
    dataloader = make_testing_dataloader(dataset, batch_size=inference_args.batch_size)
    dataloader = accelerator.prepare_data_loader(dataloader, device_placement=True)

    # Prepare model
    model = accelerator.prepare_model(model)

    # Store results
    results = []
    
    # Print evaluation method
    if accelerator.is_main_process:
        if inference_args.use_autoregressive_certainty:
            print(f"Using autoregressive certainty score calculation (max_new_tokens={inference_args.max_new_tokens})")
        else:
            print("Using teacher forcing certainty score calculation")
    
    # Evaluation loop
    for batch in tqdm(dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
        with torch.inference_mode():
            # Get model outputs
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                v_labels=batch['v_labels'],
                t_eoss=batch['t_eoss'],
                output_all_losses=True
            )

            # Get logits from backbone outputs
            backbone_outputs = model.backbone(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                return_dict=True
            )
            logits = backbone_outputs.logits

            # Normalize embeddings (consistent with training)
            text_embed_norm = outputs.text_final_embed / outputs.text_final_embed.norm(dim=-1, keepdim=True)
            image_embed_norm = outputs.image_final_embed / outputs.image_final_embed.norm(dim=-1, keepdim=True)
            
            # Calculate scores
            if inference_args.use_autoregressive_certainty:
                certainty_score = calculate_certainty_score_autoregressive(
                    model, tokenizer, batch['input_ids'], batch['attention_mask'], batch['t_eoss'], 
                    max_new_tokens=inference_args.max_new_tokens
                )
            else:
                certainty_score = calculate_certainty_score(
                    logits=logits,
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    t_eoss=batch['t_eoss']
                )
            similarity_score = calculate_similarity_score(
                text_embed_norm,  # Natural language embedding (at t_eoss position)
                image_embed_norm  # Full sequence embedding (includes formal language)
            )
            alignment_score = calculate_alignment_score(certainty_score, similarity_score)

            # Store results
            for i in range(len(batch['input_ids'])):
                # Handle both tensor and int cases
                idx_val = batch['idx1'][i]
                t_eos_val = batch['t_eoss'][i].item() if hasattr(batch['t_eoss'][i], 'item') else batch['t_eoss'][i]
                # Use v_class which contains the sequence-level label (0 or 1)
                label_val = batch['v_class'][i]
                
                results.append({
                    'idx': idx_val,
                    'question': tokenizer.decode(batch['input_ids'][i][:t_eos_val], skip_special_tokens=True),
                    'formal': tokenizer.decode(batch['input_ids'][i][t_eos_val:], skip_special_tokens=True),
                    'certainty_score': certainty_score[i].item(),
                    'similarity_score': similarity_score[i].item(),
                    'alignment_score': alignment_score[i].item(),
                    'label': label_val
                })

    # Gather results from all processes
    if accelerator.num_processes > 1:
        # For multi-GPU, we need to gather manually since accelerator.gather doesn't handle mixed types
        gathered_results = [None for _ in range(accelerator.num_processes)]
        torch.distributed.all_gather_object(gathered_results, results)
        all_results = []
        for result_list in gathered_results:
            all_results.extend(result_list)
    else:
        all_results = results

    # Save results
    if accelerator.is_main_process:
        os.makedirs(data_args.output_dir, exist_ok=True)
        output_file = os.path.join(
            data_args.output_dir,
            f"alignment_scores_{data_args.data_id}_{data_args.verifier_id}.jsonl"
        )
        
        # Calculate and print metrics
        scores = torch.tensor([r['alignment_score'] for r in all_results])
        labels = torch.tensor([r['label'] for r in all_results], dtype=torch.float32)
        certainty_scores = torch.tensor([r['certainty_score'] for r in all_results])
        similarity_scores = torch.tensor([r['similarity_score'] for r in all_results])
        
        print(f"\nTotal samples: {len(all_results)}")
        print(f"Positive samples: {labels.sum().item()}")
        print(f"Negative samples: { (len(all_results) - labels.sum().item()) }")
        
        # Calculate accuracy using different thresholds
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        print("\nAlignment Score Metrics:")
        print("Threshold | Accuracy | Precision | Recall | F1")
        print("-" * 50)
        
        best_f1 = 0.0
        best_threshold = 0.7
        
        for threshold in thresholds:
            predictions = (scores > threshold).float()
            accuracy = (predictions == labels).float().mean()
            
            # Calculate precision, recall, F1
            tp = ((predictions == 1) & (labels == 1)).sum().float()
            fp = ((predictions == 1) & (labels == 0)).sum().float()
            fn = ((predictions == 0) & (labels == 1)).sum().float()

            eps = 1e-8  # 防止除 0
            precision = torch.where((tp + fp) > 0, tp / (tp + fp + eps), torch.tensor(0.0, device=tp.device))
            recall = torch.where((tp + fn) > 0, tp / (tp + fn + eps), torch.tensor(0.0, device=tp.device))
            f1 = torch.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + eps), torch.tensor(0.0, device=tp.device))

            f1_val = f1.item()  # 防止 tensor 比较问题
            if f1_val > best_f1:
                best_f1 = f1_val
                best_threshold = threshold

            print(f"{threshold:8.1f} | {accuracy.item():8.4f} | {precision.item():9.4f} | {recall.item():6.4f} | {f1_val:6.4f}")
        
        print(f"\nBest threshold: {best_threshold} (F1: {best_f1:.4f})")
        
        # Additional analysis: score distribution
        print(f"\nScore Distribution Analysis:")
        print(f"Overall score range: [{scores.min():.4f}, {scores.max():.4f}]")
        
        # Print score distribution by label
        pos_mask = labels == 1
        neg_mask = labels == 0
        
        # Data imbalance analysis
        pos_ratio = pos_mask.sum().item() / len(labels)
        neg_ratio = neg_mask.sum().item() / len(labels)
        print(f"\nData Balance:")
        print(f"Positive samples: {pos_mask.sum().item()} ({pos_ratio:.1%})")
        print(f"Negative samples: {neg_mask.sum().item()} ({neg_ratio:.1%})")
        print(f"Class imbalance ratio: {neg_ratio/pos_ratio:.1f}:1 (neg:pos)")
        
        # Check if the model can distinguish between positive and negative samples
        if pos_mask.sum() > 0 and neg_mask.sum() > 0:
            pos_scores = scores[pos_mask]
            neg_scores = scores[neg_mask]
            
            # Statistical test (simple mean comparison)
            pos_mean = pos_scores.mean()
            neg_mean = neg_scores.mean()
            score_separation = pos_mean - neg_mean
            
            print(f"\nScore Separation Analysis:")
            print(f"Positive samples score mean: {pos_mean:.4f}")
            print(f"Negative samples score mean: {neg_mean:.4f}")
            print(f"Score separation (pos - neg): {score_separation:.4f}")
            
            if score_separation > 0:
                print("✓ Model correctly assigns higher scores to positive samples")
            else:
                print("✗ Model assigns higher scores to negative samples (potential issue)")
            
            # Score distribution percentiles
            print(f"\nScore Distribution Percentiles:")
            print(f"Positive samples: 25%={pos_scores.quantile(0.25):.4f}, 50%={pos_scores.quantile(0.5):.4f}, 75%={pos_scores.quantile(0.75):.4f}")
            print(f"Negative samples: 25%={neg_scores.quantile(0.25):.4f}, 50%={neg_scores.quantile(0.5):.4f}, 75%={neg_scores.quantile(0.75):.4f}")
            
            # Optimal threshold based on class distribution
            # For imbalanced data, we might want to use a lower threshold
            print(f"\nRecommended Analysis:")
            print(f"With {neg_ratio:.1%} negative samples, high accuracy can be achieved by predicting mostly negative")
            print(f"Focus on F1 score or balanced accuracy for better evaluation")
        
        # Print score statistics
        print(f"\nScore Statistics:")
        print(f"Alignment Score - Mean: {scores.mean():.4f}, Std: {scores.std():.4f}")
        print(f"Certainty Score - Mean: {certainty_scores.mean():.4f}, Std: {certainty_scores.std():.4f}")
        print(f"Similarity Score - Mean: {similarity_scores.mean():.4f}, Std: {similarity_scores.std():.4f}")
        
        if pos_mask.sum() > 0:
            print(f"\nPositive samples (label=1):")
            print(f"  Alignment Score - Mean: {scores[pos_mask].mean():.4f}, Std: {scores[pos_mask].std():.4f}")
            print(f"  Certainty Score - Mean: {certainty_scores[pos_mask].mean():.4f}, Std: {certainty_scores[pos_mask].std():.4f}")
            print(f"  Similarity Score - Mean: {similarity_scores[pos_mask].mean():.4f}, Std: {similarity_scores[pos_mask].std():.4f}")
        
        if neg_mask.sum() > 0:
            print(f"\nNegative samples (label=0):")
            print(f"  Alignment Score - Mean: {scores[neg_mask].mean():.4f}, Std: {scores[neg_mask].std():.4f}")
            print(f"  Certainty Score - Mean: {certainty_scores[neg_mask].mean():.4f}, Std: {certainty_scores[neg_mask].std():.4f}")
            print(f"  Similarity Score - Mean: {similarity_scores[neg_mask].mean():.4f}, Std: {similarity_scores[neg_mask].std():.4f}")

        # import matplotlib.pyplot as plt 
        # plt.hist(scores[pos_mask].numpy(), bins=50, alpha=0.5, label="Positive")
        # plt.hist(scores[neg_mask].numpy(), bins=50, alpha=0.5, label="Negative")
        # plt.legend()
        # plt.xlabel("Alignment Score")
        # plt.ylabel("Frequency")
        # plt.title("Score Distribution by Class")
        # plt.show()

        # Save detailed results
        import json
        with open(output_file, 'w') as f:
            for result in all_results:
                f.write(json.dumps(result) + '\n')
        print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main() 