# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import json
from copy import deepcopy
from typing import Dict, Optional, Union, Any

import torch
import torch.nn as nn
from torch.autograd import Variable
from swift.llm.argument import TrainArguments
from tqdm.auto import tqdm

from swift.utils import get_logger
from .trainers import Seq2SeqTrainer

logger = get_logger()


# EWC parameters are now defined in swift.trainers.arguments.Seq2SeqTrainingArguments


class EWCTrainer(Seq2SeqTrainer):
    """
    Elastic Weight Consolidation (EWC) Trainer for continual learning.
    
    This trainer implements EWC to prevent catastrophic forgetting when learning
    multiple tasks sequentially.
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
            # tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **cleaned_kwargs
        )
        
        # EWC parameters are now available in Seq2SeqTrainingArguments
        
        # EWC specific attributes
        self.fisher_information = {}
        self.previous_params = {}
        self.param_names = []
        self.gradients = {}
        self.ewc_initialized = False
        
    def _initialize_ewc(self):
        """Initialize EWC components"""
        try:
            logger.info("Initializing EWC trainer...")
            
            # Clear previous initialization if any
            self.param_names = []
            self.fisher_information = {}
            
            # Get all trainable parameters
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.param_names.append(name)
                    # Initialize Fisher information matrix as zeros
                    self.fisher_information[name] = torch.zeros_like(param.data)
                    
            # Load previous task information if available
            if self.args.task_id > 0:
                self._load_previous_task_info()
                
            logger.info(f"EWC initialized for {len(self.param_names)} parameters")
        except Exception as e:
            logger.warning(f"EWC initialization failed: {e}")
            logger.warning("Continuing with standard training...")
            self.param_names = []
            self.fisher_information = {}
        
    def _load_previous_task_info(self):
        """Load previous task parameters and Fisher information"""
        if self.args.previous_task_checkpoint:
            logger.info(f"Loading previous task checkpoint: {self.args.previous_task_checkpoint}")
            checkpoint = torch.load(self.args.previous_task_checkpoint, map_location='cpu')
            
            # Load previous parameters
            if 'model' in checkpoint:
                prev_state_dict = checkpoint['model']
            else:
                prev_state_dict = checkpoint
                
            for name in self.param_names:
                if name in prev_state_dict:
                    self.previous_params[name] = prev_state_dict[name].clone()
                    
        # Load Fisher information if available
        if self.args.fisher_save_path and os.path.exists(self.args.fisher_save_path):
            logger.info(f"Loading Fisher information: {self.args.fisher_save_path}")
            fisher_data = torch.load(self.args.fisher_save_path, map_location='cpu')
            for name in self.param_names:
                if name in fisher_data:
                    self.fisher_information[name] = fisher_data[name].to(self.args.device)
    
    def _save_grad(self, name):
        """Create hook to save gradients for Fisher information calculation"""
        def hook(grad):
            if grad is not None:
                grad = torch.nan_to_num(grad, nan=0)
                self.gradients[name] = grad.detach().clone()
        return hook
    
    def _register_hooks(self):
        """Register hooks to capture gradients"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.param_names:
                param.register_hook(self._save_grad(name))
    
    def _compute_fisher_information(self, dataloader, num_samples=None):
        """Compute Fisher information matrix using training data"""
        logger.info("Computing Fisher information matrix...")
        
        if num_samples is None:
            num_samples = self.args.fisher_sample_size
        
        # Check if using DeepSpeed
        is_deepspeed = False
        
        if is_deepspeed:
            logger.info("Detected DeepSpeed - using train mode for Fisher computation")
            self.model.train()  # DeepSpeed needs train mode for gradient computation
            # Save original parameters to restore after Fisher computation
            original_params = {}
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            for name, param in base_model.named_parameters():
                if param.requires_grad and name in self.param_names:
                    original_params[name] = param.data.clone()
        else:
            self.model.eval()
        
        # Initialize Fisher information
        for name in self.param_names:
            self.fisher_information[name].zero_()
            
        # Register hooks to capture gradients
        self._register_hooks()
        
        sample_count = 0
        batch_count = 0
        with torch.enable_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Computing Fisher")):
                if sample_count >= num_samples:
                    break
                
                # Debug: Log batch type and content
                if batch_idx == 0:
                    logger.info(f"First batch type: {type(batch)}")
                    if isinstance(batch, dict):
                        logger.info(f"First batch keys: {batch.keys()}")
                
                # Skip None or empty batches
                if batch is None:
                    logger.warning(f"Skipping None batch at index {batch_idx}")
                    continue
                    
                try:
                    # Move batch to device - preserve original for logging
                    original_batch_keys = list(batch.keys()) if isinstance(batch, dict) else None
                    batch = self._prepare_inputs(batch)
                    
                    # Detailed validation
                    if batch is None:
                        logger.warning(f"_prepare_inputs returned None for batch {batch_idx}")
                        logger.warning(f"Original batch had keys: {original_batch_keys}")
                        continue
                    
                    if not isinstance(batch, dict):
                        logger.warning(f"Batch {batch_idx} is not a dict after _prepare_inputs: {type(batch)}")
                        continue
                    
                    if 'input_ids' not in batch:
                        logger.warning(f"Batch {batch_idx} missing 'input_ids'. Keys: {batch.keys()}")
                        continue
                    
                    if batch['input_ids'] is None:
                        logger.warning(f"Batch {batch_idx} has None 'input_ids'")
                        continue
                    
                    # Forward pass
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    
                    # Backward pass to compute gradients
                    if is_deepspeed:
                        # For DeepSpeed, use its backward and step methods
                        self.model.backward(loss)
                        self.model.step()  # Required to finalize gradient computation in DeepSpeed
                    else:
                        self.model.zero_grad()
                        loss.backward()
                    
                    # Update Fisher information with squared gradients
                    for name in self.param_names:
                        if name in self.gradients:
                            self.fisher_information[name] += (self.gradients[name] ** 2) / num_samples
                    
                    # Count samples
                    sample_count += batch['input_ids'].size(0)
                    batch_count += 1
                    
                except Exception as e:
                    logger.warning(f"Error processing batch {batch_idx} for Fisher computation: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    continue
        
        # Restore original parameters for DeepSpeed after all Fisher computation
        if is_deepspeed and original_params:
            logger.info("Restoring original model parameters after Fisher computation")
            base_model = self.model.module if hasattr(self.model, 'module') else self.model
            for name, param in base_model.named_parameters():
                if param.requires_grad and name in original_params:
                    param.data.copy_(original_params[name])
                
        if batch_count == 0:
            logger.warning("No valid batches processed for Fisher computation!")
        else:
            logger.info(f"Fisher information computed using {sample_count} samples from {batch_count} batches")
        
    def _ewc_penalty(self):
        """Compute EWC regularization penalty"""
        penalty = 0.0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.param_names:
                if name in self.previous_params:
                    # EWC penalty: λ/2 * F_i * (θ_i - θ*_i)?
                    penalty += torch.sum(
                        self.fisher_information[name] * 
                        (param - self.previous_params[name].to(param.device)) ** 2
                    )
                    
        return penalty
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Override compute_loss to add EWC regularization"""
        # Initialize EWC if not done yet
        if not self.ewc_initialized:
            self._initialize_ewc()
            self.ewc_initialized = True
            
        # Compute standard loss
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        # Add EWC penalty for tasks after the first one
        try:
            if self.args.task_id > 0 and self.previous_params and len(self.param_names) > 0:
                ewc_penalty = self._ewc_penalty()
                loss += 0.5 * self.args.ewc_lambda * ewc_penalty
                
                # Log EWC loss for monitoring
                if hasattr(self, 'state') and self.state.global_step % self.args.logging_steps == 0:
                    logger.info(f"Step {self.state.global_step}: EWC penalty = {ewc_penalty.item():.6f}")
        except Exception as e:
            logger.warning(f"EWC penalty calculation failed: {e}")
            logger.warning("Continuing with standard training...")
        
        return (loss, outputs) if return_outputs else loss
    
    def train(self, *args, **kwargs):
        """Override train method to compute Fisher information after training"""
        # Initialize EWC if not done yet
        if not self.ewc_initialized:
            self._initialize_ewc()
            self.ewc_initialized = True
            
        # Standard training
        result = super().train(*args, **kwargs)
        
        # Compute Fisher information after training (for next task)
        try:
            logger.info("Attempting to compute Fisher information...")
            
            # Try to get a fresh dataloader
            train_dataloader = None
            if self.train_dataset is not None:
                logger.info("Creating fresh dataloader for Fisher computation...")
                train_dataloader = self.get_train_dataloader()
            elif hasattr(self, 'train_dataloader') and self.train_dataloader is not None:
                logger.info("Using cached train_dataloader for Fisher computation...")
                train_dataloader = self.train_dataloader
            
            if train_dataloader is not None:
                # Verify dataloader has data
                try:
                    test_batch = next(iter(train_dataloader))
                    if test_batch is None:
                        logger.warning("Dataloader returns None batches, skipping Fisher computation")
                    else:
                        logger.info(f"Dataloader verified, computing Fisher information...")
                        self._compute_fisher_information(train_dataloader)
                        # Save Fisher information and current parameters
                        self._save_task_info()
                except StopIteration:
                    logger.warning("Dataloader is empty, skipping Fisher computation")
            else:
                logger.warning("No train dataloader available, skipping Fisher computation")
                
        except Exception as e:
            logger.warning(f"Failed to compute Fisher information: {e}")
            import traceback
            logger.warning(traceback.format_exc())
        
        return result
    
    def _save_task_info(self):
        """Save current task information for next task"""
        logger.info("Saving task information for EWC...")
        
        # Save current parameters as previous parameters for next task
        current_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.param_names:
                current_params[name] = param.data.cpu().clone()
                
        # Save to checkpoint directory
        if self.args.output_dir:
            # Save Fisher information
            fisher_save_path = os.path.join(self.args.output_dir, f"fisher_task_{self.args.task_id}.pt")
            torch.save(self.fisher_information, fisher_save_path)
            logger.info(f"Fisher information saved to: {fisher_save_path}")
            
            # Save current parameters
            params_save_path = os.path.join(self.args.output_dir, f"params_task_{self.args.task_id}.pt")
            torch.save(current_params, params_save_path)
            logger.info(f"Parameters saved to: {params_save_path}")
            
            # Save task metadata
            metadata = {
                "task_id": self.args.task_id,
                "ewc_lambda": self.args.ewc_lambda,
                "fisher_sample_size": self.args.fisher_sample_size,
                "num_parameters": len(self.param_names)
            }
            metadata_path = os.path.join(self.args.output_dir, f"ewc_metadata_task_{self.args.task_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2) 