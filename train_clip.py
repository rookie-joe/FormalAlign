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
# os.environ['WANDB_PROJECT'] = "verifier-reproduce"
os.environ['WANDB_PROJECT'] = "formalalign"

from utils.states import set_deepspeed_config, set_training_states, set_random_seed
from utils.optim import get_optimizers
from utils.models import save_training_args_with_accelerator
from utils.verifier_models import save_verifier, save_verifier_checkpoint, save_best_verifier_checkpoint, build_verifier_clip
from utils.datasets import make_training_verifier_data_module, make_training_dataloaders
from utils.metrics import VerifierClassificationAcc
from utils.mma.datasets import make_finetuning_generator_data_module


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    project_dim : Optional[int] = field(default=512)
    clip_temperature: Optional[float] = field(default=0.07)

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
                loss = output.loss
                all_losses = output.all_losses
                accelerator.backward(loss)
                optimizer.step()
                # if not accelerator.optimizer_step_was_skipped and global_step % training_args.gradient_accumulation_steps == 0:
                lr_scheduler.step()
                optimizer.zero_grad()
                accelerator.wait_for_everyone()

            # training logging
            if accelerator.is_main_process:
                train_dataloader_iterator.set_postfix(epoch=cur_epoch, step=local_step, loss=loss.item(),
                                                      proj_loss=all_losses.get('proj_loss').item(), llm_loss=all_losses.get(
                        'llm_loss').item() if data_args.loss_on_llm else 0)

                if global_step % training_args.gradient_accumulation_steps :
                    wandb.log({
                        'loss': loss.item(),
                        'proj_loss': all_losses.get('proj_loss').item(),
                        'llm_loss': all_losses.get('llm_loss').item() if data_args.loss_on_llm else 0,
                        'lr': lr_scheduler.get_last_lr()[0],
                    }, step=global_step)

            # save checkpoint
            save_steps = training_args.save_steps if training_args.save_steps > 0 else 1000
            if global_step!= 0 and (global_step % training_args.gradient_accumulation_steps == 0) and (global_step // training_args.gradient_accumulation_steps ) % save_steps == 0 and global_step!= loaded_step:
                accelerator.wait_for_everyone()
                resume_dir = os.path.join(output_args.save_dir, str(global_step // training_args.gradient_accumulation_steps))
                print(f"saving model in {resume_dir} ")
                
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
                            print(f"Deleting old checkpoint: {old_dir}")
                            shutil.rmtree(old_dir)

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