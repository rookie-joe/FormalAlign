import torch
import transformers
from tqdm.auto import tqdm
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Set, Any, Union
import gc
from accelerate import Accelerator
import wandb
import os
import numpy as np
import torch.nn.functional as F
# os.environ['WANDB_PROJECT'] = "verifier-reproduce"
os.environ['WANDB_PROJECT'] = "formalalign"

from utils.states import set_deepspeed_config, set_training_states, set_random_seed
from utils.optim import get_optimizers
from utils.models import save_training_args_with_accelerator
from utils.verifier_models import save_verifier, build_verifier_clip
from utils.datasets import make_training_dataloaders
from utils.metrics import VerifierClassificationAcc, VerifierClipClassificationAcc_original
from utils.mma.datasets import make_finetuning_generator_data_module


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    project_dim : Optional[int] = field(default=512)
    clip_temperature: Optional[float] = field(default=0.07)
    contrastive_weight: Optional[float] = field(default=1.0)
    ce_weight: Optional[float] = field(default=1.0)
    use_hard_negatives: Optional[bool] = field(default=False)
    hard_negative_weight: Optional[float] = field(default=0.1)

@dataclass
class DataArguments:
    data_dir: str = field(default='data/gsm8k/model_generation', metadata={"help": "Path to the training data."})
    data_id : str = field(default='none')
    target_set: str = field(default='train')
    val_target_set: str = field(default=None)
    generator_id: str = field(default='llama7b-2-ep2')

    per_problem_sampling_solution: int = field(default=-1)
    loss_level: str = field(default='token')
    loss_on_llm: bool = field(default=False)

    dedup: bool = field(default=False)
    process: bool = field(default=False)

    verifier_id: str = field(default='llama7b-2-ep2')
    self_training: bool = field(default=False)
    easy : bool = field(default=True)

@dataclass
class TrainingArguments:
    cache_dir: Optional[str] = field(default=None)
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )

    max_steps: int = field(default=-1, metadata={"help": "When it is specified, num_train_epoches is ignored"})
    num_train_epoches: int = field(default=1)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=1)
    gradient_checkpointing: bool = field(default=True)

    eval_steps: int = field(default=-1, metadata={"help": "When it is specified, eval_epoches is ignored"})
    eval_epoches: int = field(default=1)
    max_grad_norm: int = field(default=1.0)
    per_device_eval_batch_size: int = field(default=4)
    resume_from_checkpoint: bool= field(default=False)

    learning_rate: float = field(default=1e-5)
    weight_decay: float = field(default=0)
    lr_scheduler_type: str = field(default="linear")
    warmup_steps: int = field(default=-1, metadata={"help": "When it is specified, warmup_ratio is ignored"})
    warmup_ratio: float = field(default=0)

    num_lr_epoches_fs: int = field(default=-1)
    num_lr_epoches_scatter: int = field(default=-1)

    logging_steps: int = field(default=-1, metadata={"help": "When it is specified, logging_epoches is ignored"})
    logging_epoches: int = field(default=1)

    save_steps: int = field(default=-1, metadata={"help": "When it is specified, save_epoches is ignored"})
    save_epoches: int = field(default=1)
    save_total_limit: int = field(default=3)
    save_best: bool = field(default=False)
    fp16: bool = field(default=False)
    seed: int = field(default=42)
    resume: bool = field(default=False)

@dataclass
class OutputArguments:
    logging_dir: str = field(default='wandb/')
    save_dir: str = field(default='checkpoints/')


def evaluate(model, val_dataloader, accelerator, acc_thres=0.7):
    model.eval()
    val_metric = VerifierClipClassificationAcc_original(n_data=len(val_dataloader.dataset))
    
    all_image_embeds = []
    all_text_embeds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Evaluating", disable=not accelerator.is_main_process):
            batch_input = {k: v for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels', 'v_labels', 't_eoss')}
            output = model(**batch_input)
            image_final_embed = output.image_final_embed
            text_final_embed = output.text_final_embed
            
            # Normalize embeddings
            image_final_embed = image_final_embed / image_final_embed.norm(dim=-1, keepdim=True)
            text_final_embed = text_final_embed / text_final_embed.norm(dim=-1, keepdim=True)
            
            all_image_embeds.append(image_final_embed)
            all_text_embeds.append(text_final_embed)
            all_labels.append(batch['v_labels'])
            
            val_metric(image_final_embed, text_final_embed, batch['v_labels'])
    
    # Concatenate all embeddings and labels
    all_image_embeds = torch.cat(all_image_embeds, dim=0)
    all_text_embeds = torch.cat(all_text_embeds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Calculate similarity matrix with temperature scaling
    similarity = torch.matmul(all_image_embeds, all_text_embeds.transpose(0, 1)) / model.clip_temperature
    
    # Calculate additional metrics
    metrics = {}
    
    # Basic accuracy and recall
    test_acc, test_recall = val_metric.get_metric(acc_thres)
    metrics['accuracy'] = test_acc
    metrics['recall'] = test_recall
    
    # Embedding statistics
    metrics['image_embed_norm_mean'] = torch.norm(all_image_embeds, dim=1).mean().item()
    metrics['text_embed_norm_mean'] = torch.norm(all_text_embeds, dim=1).mean().item()
    metrics['image_embed_norm_std'] = torch.norm(all_image_embeds, dim=1).std().item()
    metrics['text_embed_norm_std'] = torch.norm(all_text_embeds, dim=1).std().item()
    
    # Similarity statistics
    pos_sim = similarity[torch.arange(similarity.shape[0]), torch.arange(similarity.shape[0])]
    neg_sim = similarity[~torch.eye(similarity.shape[0], dtype=bool)]
    
    metrics['pos_sim_mean'] = pos_sim.mean().item()
    metrics['pos_sim_std'] = pos_sim.std().item()
    metrics['neg_sim_mean'] = neg_sim.mean().item()
    metrics['neg_sim_std'] = neg_sim.std().item()
    
    # Distribution metrics
    metrics['pos_sim_min'] = pos_sim.min().item()
    metrics['pos_sim_max'] = pos_sim.max().item()
    metrics['neg_sim_min'] = neg_sim.min().item()
    metrics['neg_sim_max'] = neg_sim.max().item()
    
    # Ranking metrics
    sorted_indices = torch.argsort(similarity, dim=1, descending=True)
    ranks = torch.where(sorted_indices == torch.arange(sorted_indices.shape[0]).unsqueeze(1).to(sorted_indices.device))[1] + 1
    metrics['mean_rank'] = ranks.float().mean().item()
    metrics['median_rank'] = ranks.float().median().item()
    metrics['rank_1'] = (ranks == 1).float().mean().item()
    metrics['rank_5'] = (ranks <= 5).float().mean().item()
    metrics['rank_10'] = (ranks <= 10).float().mean().item()
    
    # Calculate mean reciprocal rank (MRR)
    metrics['mrr'] = (1.0 / ranks.float()).mean().item()
    
    if accelerator.is_main_process:
        print("\nValidation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        
        wandb.log(metrics, step=accelerator.state.global_step)
    
    model.train()
    return metrics

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, OutputArguments))
    model_args, data_args, training_args, output_args = parser.parse_args_into_dataclasses()
    config_args_dict = model_args.__dict__.copy().update(dict(**data_args.__dict__, **training_args.__dict__))
    set_random_seed(training_args.seed)

    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps)

    # load model, tokenizer, and dataloader
    # set_deepspeed_config(accelerator, training_args)

    # if model_args.model_name_or_path and os.path.exists(os.path.join( model_args.model_name_or_path, 'verifier.pth')):
    #     model, tokenizer = build_verifier_self(model_args, training_args , accelerator)
    # else:
    model, tokenizer = build_verifier_clip(model_args, training_args , accelerator)
    data_module = make_finetuning_generator_data_module(tokenizer, data_args)
    train_dataloader, val_dataloader = make_training_dataloaders(data_module, training_args)
    
    # config optimizer and scheduler
    set_training_states(data_module, training_args)
    optimizer, lr_scheduler = get_optimizers(model, training_args)

    # init validation metric
    # val_metric = VerifierClassificationAcc(n_data=len(data_module['val_dataset']) if data_module['val_dataset'] is not None else 0)

    if val_dataloader is not None:
        model, train_dataloader, val_dataloader, optimizer, lr_scheduler = accelerator.prepare(model, train_dataloader, val_dataloader, optimizer,lr_scheduler )
    else:
        model, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(model, train_dataloader, optimizer, lr_scheduler)

    cur_epoch = local_step = global_step = 0
    best_acc = 0.0  # Track best accuracy

    # init wandb
    if accelerator.is_main_process:
        project_name = os.environ['WANDB_PROJECT']
        logging_dir = os.path.join(output_args.logging_dir, project_name)

        os.makedirs(logging_dir, exist_ok=True)
        # Use a valid wandb run name by extracting the basename from save_dir
        wandb_id = os.path.basename(output_args.save_dir.rstrip('/'))
        wandb.init(id=wandb_id, dir=logging_dir, config=config_args_dict)

    # training
    loaded_step = -1
    loaded_step_dir = ""
    # Potentially load in the weights and states from a previous save
    if training_args.resume_from_checkpoint:
        assert os.path.exists(output_args.save_dir)

        # 获取所有子目录
        subdirs = [d for d in os.listdir(output_args.save_dir) if os.path.isdir(os.path.join(output_args.save_dir, d))]

        # 遍历所有子目录，找到最大的步数
        for subdir in subdirs:
            try:
                # 将目录名称转换为整数步数
                step = int(subdir)
                # 更新最大步数和目录
                if step > loaded_step :
                    loaded_step = step
                    loaded_step_dir = subdir
            except ValueError:
                # 如果转换失败，忽略这个目录
                continue
        assert loaded_step
        loaded_step_dir_path = os.path.join(output_args.save_dir, loaded_step_dir)
        # 这里假设accelerator.load_state是加载模型的函数
        # 实际上你需要替换成你使用的库或方法
        # accelerator.load_state(f"models/{max_step_dir_path}")
        print(f"Model to be loaded: {loaded_step_dir_path} " )
        accelerator.load_state(loaded_step_dir_path)
        loaded_step *= training_args.gradient_accumulation_steps

    start_global_step = loaded_step

    global_step =  0
    model.train()
    while global_step < training_args.num_training_steps:
        train_dataloader_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                                         desc='Training') if accelerator.is_main_process else enumerate(
            train_dataloader)

        for local_step, batch in train_dataloader_iterator:
            if global_step < start_global_step:
                global_step += 1
                continue

            batch_input = {k: v for k, v in batch.items() if k in ('input_ids', 'attention_mask', 'labels', 'v_labels', 't_eoss')}
            # backpropagation
            with accelerator.autocast(),accelerator.accumulate(model):
                output = model(**batch_input, output_all_losses=True)
                
                # Get embeddings and normalize
                image_embeds = output.image_final_embed
                text_embeds = output.text_final_embed
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                
                # Calculate similarity matrix
                logits = torch.matmul(image_embeds, text_embeds.transpose(0, 1)) / model_args.clip_temperature
                
                # Calculate contrastive loss
                labels = torch.arange(len(logits), device=logits.device)
                contrastive_loss = (
                    F.cross_entropy(logits, labels) + 
                    F.cross_entropy(logits.transpose(0, 1), labels)
                ) / 2
                
                # Calculate cross-entropy loss if enabled
                ce_loss = output.all_losses.get('llm_loss', 0.0) if data_args.loss_on_llm else 0.0
                
                # Combine losses with weights
                loss = (
                    model_args.contrastive_weight * contrastive_loss + 
                    model_args.ce_weight * ce_loss
                )
                
                # Add hard negative mining if enabled
                if model_args.use_hard_negatives:
                    # Find hard negatives (high similarity but wrong pairs)
                    mask = torch.eye(len(logits), device=logits.device).bool()
                    neg_similarities = logits.masked_fill(mask, float('-inf'))
                    hard_negative_loss = -torch.log_softmax(neg_similarities, dim=1).mean()
                    loss += model_args.hard_negative_weight * hard_negative_loss
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accelerator.wait_for_everyone()

            # training logging
            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(
                    epoch=cur_epoch, 
                    step=local_step, 
                    loss=loss.item(),
                    contrastive_loss=contrastive_loss.item(),
                    ce_loss=ce_loss if isinstance(ce_loss, float) else ce_loss.item()
                )

                if global_step % training_args.gradient_accumulation_steps:
                    log_dict = {
                        'loss': loss.item(),
                        'contrastive_loss': contrastive_loss.item(),
                        'ce_loss': ce_loss if isinstance(ce_loss, float) else ce_loss.item(),
                        'lr': lr_scheduler.get_last_lr()[0],
                    }
                    
                    if model_args.use_hard_negatives:
                        log_dict['hard_negative_loss'] = hard_negative_loss.item()
                    
                    wandb.log(log_dict, step=global_step)

            # Validation and save checkpoint
            save_steps = training_args.save_steps if training_args.save_steps > 0 else 500
            if global_step!= 0 and (global_step % training_args.gradient_accumulation_steps == 0) and (global_step // training_args.gradient_accumulation_steps ) % save_steps == 0 and global_step!= loaded_step:
                
                accelerator.wait_for_everyone()
                
                # Run validation
                if val_dataloader is not None:
                    val_metrics = evaluate(model, val_dataloader, accelerator)
                    if accelerator.is_main_process:
                        wandb.log(val_metrics, step=global_step)
                        print("\nValidation Metrics:")
                        for k, v in val_metrics.items():
                            print(f"{k}: {v:.4f}")
                        
                        # Save best model
                        if val_metrics['accuracy'] > best_acc:
                            best_acc = val_metrics['accuracy']
                            best_model_dir = os.path.join(output_args.save_dir, 'best')
                            print(f"New best accuracy: {best_acc:.4f}, saving model to {best_model_dir}")
                            accelerator.save_state(best_model_dir)
                
                # Save current checkpoint
                resume_dir = os.path.join(output_args.save_dir, str(global_step // training_args.gradient_accumulation_steps))
                print(f"saving model in {resume_dir} ")
                # save_verifier_checkpoint(accelerator, model, tokenizer, resume_dir, global_step, training_args.save_total_limit)

                # Save current checkpoint
                accelerator.save_state(resume_dir)
                
                # Clean up old checkpoints to maintain save_total_limit
                if training_args.save_total_limit > 0:
                    import shutil
                    # Get all checkpoint directories
                    checkpoint_dirs = []
                    for d in os.listdir(output_args.save_dir):
                        dir_path = os.path.join(output_args.save_dir, d)
                        if os.path.isdir(dir_path) and d.isdigit():
                            checkpoint_dirs.append((int(d), dir_path))
                    
                    # Sort by step number and remove old checkpoints
                    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
                    if len(checkpoint_dirs) > training_args.save_total_limit:
                        for _, old_dir in checkpoint_dirs[training_args.save_total_limit:]:
                            try:
                                print(f"Deleting old checkpoint: {old_dir}")
                                shutil.rmtree(old_dir, ignore_errors=True)
                            except Exception as e:
                                print(f"Warning: Failed to delete checkpoint {old_dir}: {e}")
                                # Continue training even if deletion fails
                                continue

            global_step += 1

        cur_epoch += 1
        # if cur_epoch == 1:
        #     save_verifier_checkpoint(accelerator, model, tokenizer, output_args.save_dir, global_step,
        #                              training_args.save_total_limit)

        del train_dataloader_iterator
        gc.collect();
        accelerator.wait_for_everyone()

    accelerator.wait_for_everyone()
    save_verifier(accelerator, model, tokenizer, output_args.save_dir)
    save_training_args_with_accelerator(accelerator, training_args, output_args.save_dir)

    if accelerator.is_main_process:
        # shutil.rmtree(os.path.join(output_args.save_dir, 'resume'))
        wandb.finish()

if __name__ == "__main__":
    main()