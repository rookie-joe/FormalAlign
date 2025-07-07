#!/usr/bin/env python3
"""
Simple test script to verify autoregressive certainty score calculation.
This script can be used to compare the old teacher forcing method with the new autoregressive method.
"""

import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
import numpy as np

def test_certainty_calculation():
    """Test and compare different certainty calculation methods."""
    
    # Example data (simulate NL + FL sequence)
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Example: Natural language + Formal language
    nl_text = "Find the derivative of x^2 + 3x + 1"
    fl_text = " d/dx(x^2 + 3x + 1) = 2x + 3"
    
    # Tokenize
    nl_tokens = tokenizer.encode(nl_text, return_tensors="pt")
    fl_tokens = tokenizer.encode(fl_text, return_tensors="pt")
    
    # Combine NL + FL
    combined_tokens = torch.cat([nl_tokens, fl_tokens], dim=1)
    attention_mask = torch.ones_like(combined_tokens)
    
    # t_eoss position (where FL starts)
    t_eoss = torch.tensor([nl_tokens.shape[1]])
    
    print("=== Test Data ===")
    print(f"NL text: {nl_text}")
    print(f"FL text: {fl_text}")
    print(f"NL tokens: {nl_tokens.shape}")
    print(f"FL tokens: {fl_tokens.shape}")
    print(f"Combined tokens: {combined_tokens.shape}")
    print(f"t_eoss position: {t_eoss.item()}")
    
    # Test with dummy logits (simulating model outputs)
    vocab_size = tokenizer.vocab_size
    seq_len = combined_tokens.shape[1]
    
    # Create dummy logits with some realistic patterns
    torch.manual_seed(42)
    logits = torch.randn(1, seq_len, vocab_size) * 0.1
    
    # Make the ground truth tokens more likely (simulate a good model)
    for i in range(seq_len - 1):
        true_next_token = combined_tokens[0, i + 1]
        logits[0, i, true_next_token] += 2.0  # Boost ground truth probability
    
    print("\n=== Testing Teacher Forcing Method ===")
    teacher_forcing_score = calculate_certainty_score_teacher_forcing(
        logits, combined_tokens, attention_mask, t_eoss
    )
    print(f"Teacher forcing certainty score: {teacher_forcing_score.item():.6f}")
    
    print("\n=== Testing Autoregressive Method ===")
    autoregressive_score = calculate_certainty_score_autoregressive_dummy(
        logits, combined_tokens, attention_mask, t_eoss
    )
    print(f"Autoregressive certainty score: {autoregressive_score.item():.6f}")
    
    print("\n=== Comparison ===")
    print(f"Ratio (autoregressive / teacher_forcing): {autoregressive_score.item() / teacher_forcing_score.item():.6f}")
    print(f"Difference: {abs(autoregressive_score.item() - teacher_forcing_score.item()):.6f}")
    
    # Expected: Autoregressive should be lower than teacher forcing
    # because it doesn't have access to future tokens
    if autoregressive_score.item() < teacher_forcing_score.item():
        print("✓ Expected: Autoregressive score is lower (more realistic)")
    else:
        print("⚠ Unexpected: Autoregressive score is higher")

def calculate_certainty_score_teacher_forcing(logits, input_ids, attention_mask, t_eoss):
    """Teacher forcing method (old implementation)."""
    log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    next_tokens = input_ids[:, 1:]
    token_log_probs = torch.gather(log_probs, -1, next_tokens.unsqueeze(-1)).squeeze(-1)
    
    # Create formal mask
    batch_size = input_ids.size(0)
    seq_length = token_log_probs.size(1)
    formal_mask = torch.zeros_like(token_log_probs, dtype=torch.bool)
    
    for i in range(batch_size):
        t_eos_val = t_eoss[i].item()
        if t_eos_val < seq_length:
            formal_mask[i, t_eos_val:] = True
    
    valid_mask = attention_mask[:, 1:] * formal_mask
    seq_lengths = valid_mask.sum(dim=1)
    masked_log_probs = token_log_probs * valid_mask
    
    avg_log_probs = torch.where(
        seq_lengths > 0,
        masked_log_probs.sum(dim=1) / seq_lengths.clamp(min=1),
        torch.tensor(-10.0)
    )
    
    return torch.exp(avg_log_probs)

def calculate_certainty_score_autoregressive_dummy(logits, input_ids, attention_mask, t_eoss):
    """Autoregressive method (simplified version using pre-computed logits)."""
    batch_size = input_ids.size(0)
    certainty_scores = []
    
    for i in range(batch_size):
        t_eos_val = t_eoss[i].item()
        ground_truth_fl = input_ids[i, t_eos_val:]
        
        # Find actual FL length (excluding padding)
        fl_length = len(ground_truth_fl)
        
        log_probs = []
        
        # Simulate autoregressive generation
        for step in range(fl_length):
            # Position in the original sequence where we predict this token
            pred_pos = t_eos_val + step - 1
            
            if pred_pos >= 0 and pred_pos < logits.shape[1]:
                # Get logits at this position
                next_token_logits = logits[i, pred_pos, :]
                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                
                # Ground truth next token
                ground_truth_next_token = ground_truth_fl[step]
                
                # Record log probability
                token_log_prob = next_token_log_probs[ground_truth_next_token]
                log_probs.append(token_log_prob)
        
        if len(log_probs) == 0:
            certainty_scores.append(torch.tensor(0.001))
        else:
            avg_log_prob = torch.stack(log_probs).mean()
            certainty_score = torch.exp(avg_log_prob)
            certainty_scores.append(certainty_score)
    
    return torch.stack(certainty_scores)

if __name__ == "__main__":
    test_certainty_calculation() 