from utils.states import set_random_seed
from utils.verifier_models import load_verifier_clip
from utils.datasets import make_test_verifier_data_module, make_testing_dataloader, make_test_verifierclip_data_module
from utils.metrics import VerifierClipClassificationAcc_original, VerifierClipMPk_original, VerifierClassificationAcc, VerifierMPk_original
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
    
    verifier_output_dir: str = field(default='eval_results/gsm8k/verifier', metadata={"help": "Path to save the responses and metrics."})
    generator_metric_dir: str = field(default='eval_results/gsm8k/generator_with_verifier', metadata={"help": "Path to save the responses and metrics."})
    easy : bool = field(default=True)

@dataclass
class InferenceArguments:
    batch_size: int = field(default=1)
    seed: int = field(default=None)
    acc_thres: float = field(default=0.7)
    per_device_eval_batch_size: int = field(default=4)


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
        verifier_outputs_file, verifier_metrics_file, generator_metrics_file = get_save_files(model_args, data_args,
                                                                                              inference_args)
        dataset = make_test_verifierclip_data_module(tokenizer, data_args)

        dataloader = make_testing_dataloader(dataset, batch_size=inference_args.batch_size)

        n_question = dataset.n_question
        per_problem_sampling_solution = dataset.per_problem_sampling_solution

        # - eval_with_clip: only use clip (cosine simlarity) as alignment score
        # verifier_acc_metric = VerifierClipClassificationAcc_original(n_data=len(dataset))
        # verifier_mpk_metric = VerifierClipMPk_original(n_data=len(dataset), n_solution_per_problem=per_problem_sampling_solution)

        # - eval_with_clip+cert: use clip + cert (certainty score) as alignment score
        verifier_acc_metric = VerifierClassificationAcc(n_data=len(dataset))
        verifier_mpk_metric = VerifierMPk_original(n_data=len(dataset), n_solution_per_problem=per_problem_sampling_solution)

        dataloader = accelerator.prepare_data_loader(dataloader, device_placement=True)

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

        dataloader_iterator = tqdm(enumerate(dataloader), total=len(dataloader), desc='Evaluation') if accelerator.is_main_process else enumerate(dataloader)
        all_idxs1_list, all_idxs2_list, all_vscores_list, all_labels_list =  tuple([] for _ in range(4))
        for _, batch in dataloader_iterator:
            # filter batch to only include necessary keys
            batch_input = {k: v for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels', 'v_labels', 't_eoss')}
            # get verifier output
            with torch.inference_mode(mode=True):
                output = verifier(**batch_input)
                # v_scores = output.v_scores
                image_final_embed = output.image_final_embed
                text_final_embed = output.text_final_embed
                logits = output.logits

            # 1️⃣ Certainty score
            labels = batch['labels']  # shape: (B, T)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # log_softmax + gather target token prob
            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = torch.gather(log_probs, -1, shift_labels.unsqueeze(-1)).squeeze(-1)

            # mask padding (if there is padding_token_id)
            if tokenizer.pad_token_id is not None:
                non_pad_mask = shift_labels.ne(tokenizer.pad_token_id)
                token_log_probs = token_log_probs * non_pad_mask

                # get valid token number
                lengths = non_pad_mask.sum(dim=1).float().clamp(min=1)
            else:
                lengths = torch.ones(token_log_probs.size(0), device=token_log_probs.device) * token_log_probs.size(1)

            avg_log_probs = (token_log_probs.sum(dim=1) / lengths)  # average log-prob per sequence
            certainty_scores = torch.exp(avg_log_probs)  # exponent

            # 2️⃣ Similarity score
            cosine_sims = F.cosine_similarity(image_final_embed, text_final_embed, dim=-1)

            # 3️⃣ Alignment score
            alignment_scores = (certainty_scores + cosine_sims) / 2

            # print, store, sort these alignment_scores
            for idx1, idx2, score in zip(batch['idx1'], batch['idx2'], alignment_scores):
                print(f"idx1: {idx1.item()}, idx2: {idx2.item()}, alignment_score: {score.item():.4f}")
            verifier_outputs[idx1]['outputs'][idx2]['alignment_score'] = score.item()

            # 4️⃣ calculate scores against metrics
            # - alignment score (clip + cert)
            verifier_acc_metric(alignment_scores, batch['v_labels'])
            verifier_mpk_metric(alignment_scores, batch['v_labels'])

        # calculate verifier metrics
        test_acc, test_recall = verifier_align_metric.get_metric(inference_args.acc_thres)
        mp1 = verifier_mpk_metric.get_metric(1)

        metrics = {
            '#question': n_question,
            '#solution_per_problem': per_problem_sampling_solution,
            '#total_solutions': len(dataset),
            'accuracy': test_acc,
            'recall': test_recall,
            'mp1': mp1,
        }
        accelerator.print(metrics)


if __name__ == "__main__":
    main()

