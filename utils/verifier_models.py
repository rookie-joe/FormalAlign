from pdb import Pdb
from utils.models import wrapper_safe_save_model_with_accelerator, wrapper_save_checkpoint, wrapper_save_best_checkpoint, build_model, load_model
from utils.sampling import shift_padding_to_left_2D, shift_padding_to_right_2D, find_rightmost_notpadded_positions
from utils.constants import IGNORE_INDEX


from typing import Optional, List, Dict, Set, Any, Union, Callable, Mapping
import transformers
from transformers.generation.utils import ModelOutput
from torch import nn
import torch.nn.functional as F
import torch
import pathlib
import logging
from dataclasses import dataclass, field
from accelerate import Accelerator
import os
import re
import shutil




@dataclass
class VerifierModelProjOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    proj_hiddens: torch.FloatTensor = None
    all_losses: Optional[Dict[str, torch.FloatTensor]] = None
    seq_v_scores: Optional[Dict[str, torch.FloatTensor]] = None
    image_final_embed: Optional[Dict[str, torch.FloatTensor]] = None
    text_final_embed: Optional[Dict[str, torch.FloatTensor]] = None
    



class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, text_final_embed, image_final_embed, additional_negatives=None):
        batch_size = text_final_embed.size(0)
        
        # Normalize embeddings for stability
        text_final_embed = F.normalize(text_final_embed, dim=-1)
        image_final_embed = F.normalize(image_final_embed, dim=-1)

        # Initialize logits with in-batch negatives
        logits = torch.matmul(text_final_embed, image_final_embed.t()) / self.temperature

        if additional_negatives is not None:
            # additional_negatives should be a dict containing:
            # - 'hard_negative_embeds': tensor of shape (batch_size, n_hard_negs, embed_dim)
            # - 'generated_negative_embeds': tensor of shape (batch_size, n_gen_negs, embed_dim) 
            # - 'structural_negative_embeds': tensor of shape (batch_size, n_struct_negs, embed_dim)
            
            all_negative_embeds = []
            
            for neg_type, neg_embeds in additional_negatives.items():
                # Normalize negative embeddings
                neg_embeds = F.normalize(neg_embeds, dim=-1)
                
                # Compute similarity with negative samples
                neg_logits = torch.matmul(text_final_embed, neg_embeds.transpose(-2, -1)) / self.temperature
                
                # Concatenate with main logits
                logits = torch.cat([logits, neg_logits], dim=-1)
                
                all_negative_embeds.append(neg_embeds)

        # Compute the target (positive samples are the diagonal elements)
        targets = torch.arange(batch_size).to(text_final_embed.device)

        # Calculate cross entropy loss
        loss = F.cross_entropy(logits, targets)

        return loss

class Verifier_Clip(nn.Module):

    def __init__(self, backbone, checkpoint_dir=None, clip_temperature = 0.07,project_dim =  512, torch_dtype = torch.bfloat16):
        super(Verifier_Clip, self).__init__()
        self.backbone = backbone
        self.gain = nn.Parameter(torch.randn(1,))
        self.bias = nn.Parameter(torch.randn(1,))
        self.dropout = nn.Dropout(p=0.2)
        # self.project_head = nn.Linear(self.backbone.get_input_embeddings().embedding_dim, self.project_dim, bias=False )
        
        embed_dim = self.backbone.get_input_embeddings().embedding_dim
        self.project_dim = project_dim
        
        if self.project_dim == embed_dim:
            # If project_dim matches embed_dim, use an identity mapping
            self.project_head = nn.Identity()
            print('we do not use project here')
        else:
            # Otherwise, use a linear layer for projection
            self.project_head = nn.Linear(embed_dim, self.project_dim, bias=False)

        self.loss_fn = ContrastiveLoss(clip_temperature)

        if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, 'verifier.pth')):
            print("load verifier checkpoint in the existing dir")
            verifier_params = torch.load(os.path.join(checkpoint_dir, 'verifier.pth'))
            self.load_state_dict(verifier_params, strict=False)
        else:
            print("jusing the rand")
            # self.init_head_params()

        self.to(dtype=torch_dtype)

        self.pad_token_id = backbone.config.pad_token_id

    # def init_head_params(self):
    #     # todo change
    #     output_embeddings = self.backbone.get_output_embeddings().weight.data
    #     output_embeddings_avg = output_embeddings.mean(dim=0, keepdim=True)

    #     self.project_head.weight = nn.Parameter(output_embeddings_avg)

    def loss_fct(self, v_scores: torch.FloatTensor, v_labels: torch.LongTensor):
        # (batch_size, n_seq, 1)

        return mse_loss_with_mask(v_scores.squeeze(), v_labels.type_as(v_scores))

    def transform(self, last_hidden_states):
        return self.gain * last_hidden_states + self.bias

    def forward(self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        v_labels: Optional[torch.LongTensor] = None,
        t_eoss: Optional[torch.LongTensor] = None,
        output_all_losses: Optional[bool] = None,
        additional_negatives: Optional[Dict[str, torch.Tensor]] = None,
    ):
        outputs = self.backbone(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            labels=labels, 
            use_cache=False,
            output_hidden_states=True, 
            return_dict=True,
        )

        llm_logits = outputs.logits
        llm_loss = outputs.loss
        llm_hidden_states = outputs.hidden_states

        # Get last layer hidden states
        proj_hidden_states = self.transform(llm_hidden_states[-1])
        bsz, n_seq, _ = proj_hidden_states.shape
        # Project to proj_dim dimension
        proj_output = self.project_head(self.dropout(proj_hidden_states))

        # Extract final embeddings for text and image inputs
        text_final_embed = proj_output[torch.arange(bsz), t_eoss.squeeze(), :]

        '''
        # Get NL+FL representation from last non-padding position
        index = ((n_seq - 1) - attention_mask.flip(dims=[1]).float().argmax(1)).view(-1, 1)
        image_final_embed = proj_output[torch.arange(bsz), index.squeeze(), :]
        '''
        # Extract FL embedding from FL tokens only (after t_eoss)
        # This ensures FL representation doesn't contain NL token information
        # but can still attend to NL through self-attention
        fl_embeds = []
        for i in range(bsz):
            t_eos_val = t_eoss[i].item() if hasattr(t_eoss[i], 'item') else t_eoss[i]
            # Get FL tokens (from t_eoss+1 to last non-padding token)
            fl_start = t_eos_val + 1
            # Find last non-padding position for this sample
            last_pos = ((n_seq - 1) - attention_mask[i].flip(dims=[0]).float().argmax(0)).item()
            
            if fl_start <= last_pos:
                # Take mean of FL token embeddings
                fl_embed = proj_output[i, fl_start:last_pos+1, :].mean(dim=0)
            else:
                # Fallback: if no FL tokens, use the token right after t_eoss
                fl_embed = proj_output[i, min(fl_start, n_seq-1), :]
            
            fl_embeds.append(fl_embed)
        
        image_final_embed = torch.stack(fl_embeds, dim=0)

        # Process additional negatives if provided
        processed_negatives = None
        if additional_negatives is not None:
            processed_negatives = {}
            for neg_type, neg_data in additional_negatives.items():
                # neg_data should contain input_ids and attention_mask for negative samples
                neg_outputs = self.backbone(
                    input_ids=neg_data['input_ids'],
                    attention_mask=neg_data['attention_mask'],
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
                neg_hidden = self.transform(neg_outputs.hidden_states[-1])
                neg_proj = self.project_head(self.dropout(neg_hidden))
                
                # Extract FL embeddings for negatives using same logic
                neg_fl_embeds = []
                neg_bsz, neg_seq, _ = neg_proj.shape
                for i in range(neg_bsz):
                    # Assume negatives also have t_eoss information
                    neg_t_eos = neg_data.get('t_eoss', torch.zeros(neg_bsz, dtype=torch.long))
                    t_eos_val = neg_t_eos[i].item() if hasattr(neg_t_eos[i], 'item') else neg_t_eos[i]
                    fl_start = t_eos_val + 1
                    last_pos = ((neg_seq - 1) - neg_data['attention_mask'][i].flip(dims=[0]).float().argmax(0)).item()
                    
                    if fl_start <= last_pos:
                        fl_embed = neg_proj[i, fl_start:last_pos+1, :].mean(dim=0)
                    else:
                        fl_embed = neg_proj[i, min(fl_start, neg_seq-1), :]
                    
                    neg_fl_embeds.append(fl_embed)
                
                processed_negatives[neg_type] = torch.stack(neg_fl_embeds, dim=0)

        proj_loss = self.loss_fn(text_final_embed, image_final_embed, processed_negatives)

        loss = proj_loss + (llm_loss if labels is not None else 0)

        all_losses = None
        if output_all_losses:
            all_losses = {'llm_loss': llm_loss, 'proj_loss': proj_loss}

        return VerifierModelProjOutput(
            loss=loss,
            all_losses=all_losses,
            text_final_embed=text_final_embed,
            image_final_embed=image_final_embed,
        )

    @torch.inference_mode(mode=True)
    def scoring_sequences(self, input_ids: torch.LongTensor):
        input_ids = shift_padding_to_right_2D(input_ids, pad_value=self.pad_token_id)
        outputs = self(
            input_ids=input_ids,
            attention_mask=input_ids.ne(self.pad_token_id),
        )
        inds = find_rightmost_notpadded_positions(input_ids, pad_value=self.pad_token_id)
        return outputs.v_scores[:, :, -1].gather(1, inds.view(-1, 1)).squeeze(-1)

    def gradient_checkpointing_enable(self):
        self.backbone.gradient_checkpointing_enable()
    
    def gradient_checkpointing_disable(self):
        self.backbone.gradient_checkpointing_disable()

def mse_loss_with_mask(scores: torch.FloatTensor, labels: torch.FloatTensor):
    scores = torch.where(labels.ne(IGNORE_INDEX), scores, 0)
    labels = torch.where(labels.ne(IGNORE_INDEX), labels, 0)
    return F.mse_loss(scores, labels, reduction='sum') / scores.shape[0]



@wrapper_safe_save_model_with_accelerator
def save_verifier(accelerator: Accelerator,
                  model: transformers.AutoModelForCausalLM,
                  cpu_state_dict: Mapping,
                  output_dir: str):
    cpu_state_dict_backbone = {
        k.split('backbone.')[1]: v
        for k, v in cpu_state_dict.items() if k.startswith('backbone')
    }
    cpu_state_dict_verifier = {
        k: v
        for k, v in cpu_state_dict.items() if not k.startswith('backbone')
    }
    accelerator.unwrap_model(model).backbone.save_pretrained(
        output_dir,
        state_dict=cpu_state_dict_backbone,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    accelerator.save(cpu_state_dict_verifier, os.path.join(output_dir, 'verifier.pth'))


@wrapper_save_checkpoint(save_func=save_verifier)
def save_verifier_checkpoint(accelerator: Accelerator,

                            model: transformers.AutoModelForCausalLM,
                            tokenizer: transformers.PreTrainedTokenizer,
                            checkpoint_output_dir: str):    
    ...


@wrapper_save_best_checkpoint(save_checkpoint_func=save_verifier_checkpoint)
def save_best_verifier_checkpoint(accelerator: Accelerator,
                                model: transformers.AutoModelForCausalLM,
                                tokenizer: transformers.PreTrainedTokenizer,
                                output_dir: str,
                                global_step: int,
                                save_total_limit: int=None):    
    ...







def build_verifier_clip(model_args: dataclass, training_args: dataclass, accelerator: Accelerator):
    backbone, tokenizer = build_model(model_args, training_args, accelerator)
    return Verifier_Clip(backbone, checkpoint_dir=model_args.model_name_or_path,clip_temperature = model_args.clip_temperature, project_dim = model_args.project_dim ).to(accelerator.device), tokenizer






def load_verifier_clip(model_args: dataclass):
    backbone, tokenizer = load_model(model_args)
    return Verifier_Clip(backbone, checkpoint_dir=model_args.model_name_or_path), tokenizer


def load_generator_and_verifier(model_args: dataclass):
    generator, tokenizer = load_model(model_args)

    v_backbone = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.verifier_model_name_or_path,
        torch_dtype=torch.float16 if model_args.fp16 else torch.bfloat16,
    )

    verifier = Verifier(v_backbone, checkpoint_dir=model_args.verifier_model_name_or_path)
    return generator, verifier, tokenizer



