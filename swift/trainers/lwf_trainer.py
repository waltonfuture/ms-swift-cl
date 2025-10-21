# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import json
from copy import deepcopy
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from swift.llm.argument import TrainArguments
from tqdm.auto import tqdm
from safetensors.torch import load_file
from swift.utils import get_logger
from .trainers import Seq2SeqTrainer

logger = get_logger()


class LwFTrainer(Seq2SeqTrainer):
    """
    Learning without Forgetting (LwF) Trainer for continual learning.
    
    This trainer implements LwF following TRACE-master's approach:
    - Load and freeze the previous task model during training
    - For each batch, compute previous model logits in real-time
    - Add KL divergence loss between current and previous model outputs
    - Ensures perfect batch matching and memory efficiency
    """
    
    def __init__(self, 
                 model=None,
                 args=None, 
                 data_collator=None,
                 train_dataset=None,
                 eval_dataset=None,
                 tokenizer=None,
                 model_init=None,
                 compute_metrics=None,
                 callbacks=None,
                 optimizers=(None, None),
                 preprocess_logits_for_metrics=None,
                 **kwargs):
        
        # Remove any conflicting parameters from kwargs to avoid duplicate arguments
        conflicting_params = ['model', 'args', 'data_collator', 'train_dataset', 'eval_dataset', 
                             'tokenizer', 'model_init', 'compute_metrics', 'callbacks', 
                             'optimizers', 'preprocess_logits_for_metrics']
        cleaned_kwargs = {k: v for k, v in kwargs.items() if k not in conflicting_params}
        
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **cleaned_kwargs
        )
        
        # LwF specific attributes (TRACE-master style)
        self.previous_model = None  # Store previous model for real-time logits computation
        self.lwf_initialized = False
        self.temperature = getattr(self.args, 'lwf_temperature', 2.0)  # Temperature for knowledge distillation
        
    def _initialize_lwf(self):
        """Initialize LwF components following TRACE-master's approach"""
        try:
            logger.info("Initializing LwF trainer (TRACE-master style)...")
            
            # For first task (task_id == 0), no previous model needed
            if self.args.task_id == 0:
                logger.info("First task: no previous model needed")
                self.lwf_initialized = True
                return
            
            # For subsequent tasks, load previous model for real-time logits computation
            logger.info(f"Task {self.args.task_id}: loading previous model...")
            self.previous_model = self._load_previous_model()
            
            if self.previous_model is not None:
                # Set previous model to eval mode and freeze parameters
                self.previous_model.eval()
                for param in self.previous_model.parameters():
                    param.requires_grad = False
                logger.info("Previous model loaded and frozen successfully")
            else:
                logger.warning("Failed to load previous model - continuing without LwF")
                self.lwf_initialized = True
                return
            
            logger.info(f"LwF initialized for task {self.args.task_id}")
            logger.info(f"LwF lambda: {self.args.lwf_lambda}, Temperature: {self.temperature}")
            
        except Exception as e:
            logger.warning(f"LwF initialization failed: {e}")
            logger.warning("Continuing with standard training...")
            self.previous_model = None
        
    def _compute_previous_logits_for_batch(self, inputs):
        """
        Compute previous model logits for current batch in real-time
        """
        if self.previous_model is None:
            return None
            
        with torch.no_grad():
            # Ensure previous model is in eval mode
            self.previous_model.eval()
            
            # Forward pass with previous model
            previous_outputs = self.previous_model(**inputs)
            
            # Return logits in float32 for stable computation
            return previous_outputs.logits.float()
    
    # def _load_previous_model(self):
    #     """Load the previous task model"""
    #     try:
    #         logger.info(f"Loading previous model from: {self.args.previous_task_checkpoint}")
            
    #         # Create a copy of current model structure
    #         previous_model = deepcopy(self.model)
            
    #         # Load checkpoint
    #         if os.path.isdir(self.args.previous_task_checkpoint):
    #             checkpoint_dir = self.args.previous_task_checkpoint
    #             # 检查是否存在分块模型的索引文件
    #             index_file = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    #             if os.path.exists(index_file):
    #                 # 加载分块模型
    #                 with open(index_file, 'r', encoding='utf-8') as f:
    #                     index_data = json.load(f)
                    
    #                 checkpoint = {}
    #                 # 遍历所有分块文件
    #                 for shard_name in index_data['weight_map'].values():
    #                     shard_path = os.path.join(checkpoint_dir, shard_name)
    #                     if not os.path.exists(shard_path):
    #                         logger.error(f"Missing shard file: {shard_path}")
    #                         return None
    #                     # 加载当前分块并合并到checkpoint中
    #                     shard = load_file(shard_path)
    #                     checkpoint.update(shard)
    #             # 检查是否是单文件safetensors模型
    #             elif os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")):
    #                 checkpoint = load_file(os.path.join(checkpoint_dir, "model.safetensors"))
    #             # 检查是否是单文件pytorch模型
    #             elif os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")):
    #                 checkpoint = torch.load(os.path.join(checkpoint_dir, "pytorch_model.bin"), map_location='cpu')
    #             else:
    #                 logger.error(f"No valid model files found in {checkpoint_dir}")
    #                 return None
    #         else:
    #             # 处理单个模型文件（非目录）
    #             checkpoint = torch.load(self.args.previous_task_checkpoint, map_location='cpu')
            
    #         # Extract state dict
    #         if isinstance(checkpoint, dict) and 'model' in checkpoint:
    #             state_dict = checkpoint['model']
    #         else:
    #             state_dict = checkpoint
            
    #         # Load state dict into previous model
    #         previous_model.load_state_dict(state_dict, strict=False)
    #         previous_model.to(self.args.device)
            
    #         logger.info("Previous model loaded successfully")
    #         return previous_model
            
    #     except Exception as e:
    #         logger.error(f"Failed to load previous model: {e}")
    #         import traceback
    #         logger.error(traceback.format_exc())
    #         return None
    def _load_previous_model(self):
        """Load the previous task model - simplified version"""
        try:
            from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM
            
            previous_checkpoint = self.args.previous_task_checkpoint
            
            # Determine model class based on current model
            if hasattr(self.model, 'config'):
                model_class = type(self.model)
            else:
                # Fallback to AutoModel
                model_class = AutoModelForCausalLM if hasattr(self.model, 'lm_head') else AutoModelForSeq2SeqLM
            
            logger.info(f"Loading previous model using {model_class.__name__}")
            
            # Load model using transformers library
            previous_model = model_class.from_pretrained(
                previous_checkpoint,
                torch_dtype=torch.bfloat16,  # 使用float32确保稳定性
                trust_remote_code=True,
                device_map=None  # 我们将手动管理设备
            ).eval()
            
            # Move to appropriate device
            previous_model.to(self.args.device)
            logger.info("Previous model loaded successfully")
            return previous_model
            
        except Exception as e:
            logger.error(f"Failed to load previous model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _compute_kd_loss(self, current_logits, previous_logits, temperature=2.0):
        """
        Compute Knowledge Distillation loss (TRACE-master style)
        Using KL divergence between softened distributions
        """
        # Since we're computing previous logits in real-time with the same inputs,
        # the shapes should match exactly. But add a safety check just in case.
        if current_logits.shape != previous_logits.shape:
            logger.warning(f"Shape mismatch - Current: {current_logits.shape}, Previous: {previous_logits.shape}")
            # Take minimum dimensions for safety
            min_seq_len = min(current_logits.shape[1], previous_logits.shape[1])
            min_vocab_size = min(current_logits.shape[2], previous_logits.shape[2])
            current_logits = current_logits[:, :min_seq_len, :min_vocab_size]
            previous_logits = previous_logits[:, :min_seq_len, :min_vocab_size]
            logger.warning(f"Aligned to shape: {current_logits.shape}")
        
        # Apply temperature scaling and compute KL divergence
        # Following TRACE-master: F.kl_div(log_softmax(prev/T), softmax(curr/T))
        # kd_loss = F.kl_div(
        #     F.log_softmax(previous_logits / temperature, dim=-1),
        #     F.softmax(current_logits / temperature, dim=-1),
        #     reduction='batchmean'
        # )
    
        kd_loss = F.kl_div(
            F.log_softmax(current_logits / temperature, dim=-1),
            F.softmax(previous_logits / temperature, dim=-1),
            reduction='batchmean'
        )
        return kd_loss
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to add LwF regularization (TRACE-master style)
        """
        # Initialize LwF if not done yet
        if not self.lwf_initialized:
            self._initialize_lwf()
            self.lwf_initialized = True
        
        # Compute standard loss
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        # Add KD loss for tasks after the first one (TRACE-master: if i_task != 0)
        if self.args.task_id > 0 and self.previous_model is not None:
            try:
                # Compute previous model logits for current batch
                previous_logits = self._compute_previous_logits_for_batch(inputs)
                
                if previous_logits is not None:
                    # Compute KD loss using current batch
                    kd_loss = self._compute_kd_loss(
                        outputs.logits, 
                        previous_logits, 
                        self.temperature
                    )
                    
                    # Add to total loss (TRACE-master: loss += KD_loss)
                    loss = loss + self.args.lwf_lambda * kd_loss
                    
                    # Log KD loss periodically
                    if hasattr(self, 'state') and self.state.global_step % self.args.logging_steps == 0:
                        logger.info(f"Step {self.state.global_step}: KD loss = {kd_loss.item():.6f}, LwF lambda = {self.args.lwf_lambda}, Total loss = {loss.item():.6f}")
                        
            except Exception as e:
                logger.warning(f"KD loss calculation failed: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")
        
        return (loss, outputs) if return_outputs else loss
    
    def train(self, *args, **kwargs):
        """
        Override train method to initialize LwF before training
        """
        # Initialize LwF if not done yet
        if not self.lwf_initialized:
            self._initialize_lwf()
            self.lwf_initialized = True
        
        # Standard training
        logger.info(f"Starting LwF training for task {self.args.task_id}...")
        result = super().train(*args, **kwargs)
        
        # After training, clean up previous model to free memory
        if self.previous_model is not None:
            del self.previous_model
            self.previous_model = None
            torch.cuda.empty_cache()
            logger.info("Previous model cleaned up to free memory")
        
        logger.info(f"LwF training completed for task {self.args.task_id}")
        
        return result 