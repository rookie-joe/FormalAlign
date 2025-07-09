from utils.states import set_random_seed
from utils.verifier_models import load_verifier_clip
from utils.datasets import make_test_verifier_data_module, make_testing_dataloader, make_test_verifierclip_data_module
from utils.metrics import VerifierClipClassificationAcc_original, VerifierClipMPk_original, VerifierClassificationAcc, VerifierMPk_original, AlignmentMetric, AlignmentMPk
from accelerate import Accelerator
from torch.nn import functional as F


import torch
import transformers
from dataclasses import dataclass, field
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from typing import Optional, List, Dict, Set, Any, Union
import os
import json
import pandas as pd
import gc


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    fp16: Optional[bool] = field(default=False)
    project_dim: Optional[int] = field(default=512)

@dataclass
class DataArguments:
    data_dir: str = field(default='data/gsm8k/model_generation', metadata={"help": "Path to the training data."})
    target_set: str = field(default='test', metadata={"help": "specify which data set to generate"})
    generator_id: str = field(default='llama7b-2-ep2')
    data_id: str = field(default='test')
    verifier_id: str = field(default='default')
    per_problem_sampling_solution: int = field(default=5)
    verifier_output_dir: str = field(default='eval_results/gsm8k/verifier', metadata={"help": "Path to save the responses and metrics."})
    generator_metric_dir: str = field(default='eval_results/gsm8k/generator_with_verifier', metadata={"help": "Path to save the responses and metrics."})
    easy : bool = field(default=True)
    loss_on_llm: bool = field(default=True)

@dataclass
class InferenceArguments:
    batch_size: int = field(default=1)
    seed: int = field(default=None)
    acc_thres: float = field(default=0.7)
    per_device_eval_batch_size: int = field(default=16)


def get_save_files(model_args: dataclass, data_args: dataclass, inference_args: dataclass):
    verifier_output_dir = os.path.join(data_args.verifier_output_dir, data_args.target_set)
    generator_metric_dir = os.path.join(data_args.generator_metric_dir, data_args.target_set)

    verifier_id = os.path.basename(os.path.normpath(model_args.model_name_or_path))
    verifier_id_short = verifier_id[:100]
    generator_metric_dir = os.path.join(generator_metric_dir, verifier_id_short)
    os.makedirs(generator_metric_dir, exist_ok=True)

    generator_id_suffix = f"_g({data_args.generator_id[:10]})"  # Shortening generator ID
    verifier_id_suffix = f"_v({verifier_id_short})"

    verifier_suffix = (verifier_id_suffix + generator_id_suffix).lstrip('_')
    generator_suffix = (generator_id_suffix + verifier_id_suffix).lstrip('_')

    verifier_outputs_file = f"responses_{verifier_suffix}.jsonl"
    verifier_metrics_file = f"metrics_{verifier_suffix}.json"
    generator_metrics_file = f"metrics_{generator_suffix}.csv"
    return os.path.join(verifier_output_dir, verifier_outputs_file), os.path.join(verifier_output_dir, verifier_metrics_file), os.path.join(generator_metric_dir, generator_metrics_file)


def extract_sol_vscores(qns_tokens: List[List[int]], sols_tokens: List[List[int]], batch_vscores: torch.FloatTensor, batch_vlabels:  torch.FloatTensor) -> List[list]:
    sol_vscores = []
    raw_vscores = []
    for qn_tokens, sol_tokens, vscores, vlabels in zip(qns_tokens, sols_tokens, batch_vscores, batch_vlabels):
        svs = vscores[len(qn_tokens):len(qn_tokens)+len(sol_tokens)+1][:, 0]
        sol_vscores.append(svs.tolist()) # (padded_response_len, )
        raw_vscores.append((vscores[:, 0].tolist(), vlabels.tolist()))  # (padded_response_len, )
    return sol_vscores, raw_vscores
    

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, InferenceArguments))
    model_args, data_args, inference_args = parser.parse_args_into_dataclasses()
    if inference_args.seed is not None:
        set_random_seed(inference_args.seed)

    accelerator = Accelerator()
    verifier, tokenizer = load_verifier_clip(model_args)

    generator_id_list = data_args.generator_id.split(",")
    for item in generator_id_list:
        data_args.generator_id = item

        # ⬇️ prepare save files, dataset, dataloader
        verifier_outputs_file, verifier_metrics_file, generator_metrics_file = get_save_files(
            model_args, data_args, inference_args
        )
        dataset = make_test_verifierclip_data_module(tokenizer, data_args)
        dataloader = make_testing_dataloader(dataset, batch_size=inference_args.batch_size)
        dataloader = accelerator.prepare_data_loader(dataloader, device_placement=True)

        # ⬇️ re-init metric inside loop (avoid accumulating results)
        n_question = dataset.n_question
        per_problem_sampling_solution = dataset.per_problem_sampling_solution
        verifier_acc_metric = AlignmentMetric(n_data=len(dataset))
        verifier_mpk_metric = AlignmentMPk(n_data=len(dataset), n_solution_per_problem=per_problem_sampling_solution)

        # prepare verifier_outputs
        verifier_outputs = []
        for data in dataset:
            if len(verifier_outputs) == 0 or verifier_outputs[-1]['idx'] != data['idx1']:
                verifier_outputs.append({
                    'idx': data['idx1'],
                    'question': data['qn_str'],
                    'outputs': [],
                })
            verifier_outputs[-1]['outputs'].append({
                'response': data['sol_str'],
                'tokens': data['sol_tokens'],
                'label': data['v_class'],
            })

        verifier.eval().cuda()
        accelerator.unwrap_model(verifier).gradient_checkpointing_enable()
        accelerator.wait_for_everyone()

        dataloader_iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc='Evaluation') \
                            if accelerator.is_main_process else enumerate(dataloader)

        for _, batch in dataloader_iterator:
            # v_class是groundtruth bool, 而labels是next token prediction
            batch_input = {k: v for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels', 'v_labels', 't_eoss')}

            with torch.inference_mode():
                output = verifier(**batch_input)
                image_final_embed = output.image_final_embed
                text_final_embed = output.text_final_embed
                logits = output.logits

            # Certainty Score
            labels = batch['labels'] # here labels mean causal token labels for token prediction (B, seq_len)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            safe_shift_labels = shift_labels.clone()
            safe_shift_labels[safe_shift_labels == -100] = 0
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(log_probs, -1, safe_shift_labels.unsqueeze(-1)).squeeze(-1)
            if tokenizer.pad_token_id is not None:
                non_pad_mask = (shift_labels != tokenizer.pad_token_id) & (shift_labels != -100)
            else:
                non_pad_mask = (shift_labels != -100)
            token_log_probs = token_log_probs * non_pad_mask
            lengths = non_pad_mask.sum(dim=1).float().clamp(min=1)
            avg_log_probs = token_log_probs.sum(dim=1) / lengths
            certainty_scores = torch.exp(avg_log_probs)

            # Similarity Score
            cosine_sims = F.cosine_similarity(image_final_embed, text_final_embed, dim=-1)

            # Alignment Score
            alignment_scores = (certainty_scores + cosine_sims) / 2
            alignment_scores = alignment_scores.view(-1, 1)

            for idx1, idx2, score in zip(batch['idx1'], batch['idx2'], alignment_scores):
                print(f"idx1: {idx1}, idx2: {idx2}, alignment_score: {score.item():.4f}")
                verifier_outputs[idx1]['outputs'][idx2]['alignment_score'] = score.item()

            # calculate metric against groundtruth labels
            verifier_acc_metric(alignment_scores, batch['v_class'])
            verifier_mpk_metric(alignment_scores, batch['v_class'])


        # ✅ 只 gather 当前 eval 得到的 scores，不要历史累积
        local_scores = np.array(verifier_acc_metric.scores)
        local_gts = np.array(verifier_acc_metric.gts)

        scores_tensor = torch.from_numpy(local_scores).float().to(accelerator.device)
        gts_tensor = torch.from_numpy(local_gts).float().to(accelerator.device)

        gathered_scores = accelerator.gather_for_metrics(scores_tensor).cpu().numpy()
        gathered_gts = accelerator.gather_for_metrics(gts_tensor).cpu().numpy()

        print(f"[DEBUG] Gathered scores size: {gathered_scores.shape}, gts size: {gathered_gts.shape}")

        # ✅ 用 gathered 数据计算 metric，**不再用 verifier_acc_metric.scores 里的历史数据**
        pred = gathered_scores > inference_args.acc_thres
        corrs = np.where(gathered_gts.astype(bool), pred, ~pred)

        if len(corrs) == 0:
            acc, precision, recall, f1 = 0.0, 0.0, 0.0, 0.0
        else:
            acc = np.sum(corrs) / len(corrs)
            tp = np.sum((gathered_gts.astype(bool)) & pred)
            fn = np.sum((gathered_gts.astype(bool)) & (~pred))
            fp = np.sum((~gathered_gts.astype(bool)) & pred)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mpk = verifier_mpk_metric.get_metric(k=1)

        # ✅ Reset metric buffers (清空避免下轮重复累积)
        verifier_acc_metric.scores = []
        verifier_acc_metric.gts = []
        verifier_mpk_metric.scores = []
        verifier_mpk_metric.gts = []

        # ✅ 打印
        metrics = {
            '#question': n_question,
            '#solution_per_problem': per_problem_sampling_solution,
            '#total_solutions': len(dataset),
            'accuracy': acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mp1': mpk,
        }
        accelerator.print(metrics)


        
if __name__ == "__main__":
    main()

