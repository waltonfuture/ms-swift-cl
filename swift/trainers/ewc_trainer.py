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
    
    This trainer implements EWC following TRACE-master's approach:
    - Fisher information is accumulated during training (not computed after)
    - Only stores previous task's parameters and Fisher information
    - EWC penalty protects only the immediate previous task
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
        
        # EWC specific attributes (TRACE-master style)
        self.fisher = {}  # Fisher information from previous task (will be overwritten each task)
        self.previous_params = {}  # Parameters from previous task (will be overwritten each task)
        self.grads = {}  # Store gradients during training
        self.param_names = []  # List of parameter names to track
        self.train_dataset_length = 0  # Total number of batches for normalization
        self.ewc_initialized = False
        self.hooks_registered = False
        
    def _initialize_ewc(self):
        """Initialize EWC components following TRACE-master's approach"""
        try:
            logger.info("Initializing EWC trainer (TRACE-master style)...")
            
            # Clear previous initialization if any
            self.param_names = []
            self.fisher = {}
            
            # Get all trainable parameters and initialize Fisher to zeros
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.param_names.append(name)
                    # Initialize Fisher information as zeros (keep on GPU for efficiency)
                    self.fisher[name] = torch.zeros_like(param.data)
            
            # Load previous task information if this is not the first task
            if self.args.task_id > 0:
                self._load_previous_task_info()
            else:
                # For first task, initialize previous_params with current model params
                logger.info("First task: initializing previous_params with current model parameters")
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.param_names:
                        self.previous_params[name] = param.data.clone()
                        
            logger.info(f"EWC initialized for {len(self.param_names)} parameters")
            logger.info(f"Task ID: {self.args.task_id}, EWC lambda: {self.args.ewc_lambda}")
            
        except Exception as e:
            logger.warning(f"EWC initialization failed: {e}")
            logger.warning("Continuing with standard training...")
            self.param_names = []
            self.fisher = {}
        
    def _load_previous_task_info(self):
        """Load previous task parameters and Fisher information"""
        # Load Fisher information from previous task
        if self.args.fisher_save_path and os.path.exists(self.args.fisher_save_path):
            logger.info(f"Loading Fisher information from: {self.args.fisher_save_path}")
            try:
                fisher_data = torch.load(self.args.fisher_save_path, map_location=self.model.device)
                for name in self.param_names:
                    if name in fisher_data:
                        self.fisher[name] = fisher_data[name].to(self.model.device)
                logger.info("Fisher information loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load Fisher information: {e}")
        else:
            logger.warning(f"Fisher information file not found: {self.args.fisher_save_path}")
            
        # Load previous task checkpoint
        if self.args.previous_task_checkpoint:
            logger.info(f"Loading previous task checkpoint: {self.args.previous_task_checkpoint}")
            try:
                # Try to load the checkpoint
                if os.path.isdir(self.args.previous_task_checkpoint):
                    # It's a directory, look for pytorch_model.bin or model.safetensors
                    checkpoint_file = None
                    if os.path.exists(os.path.join(self.args.previous_task_checkpoint, "pytorch_model.bin")):
                        checkpoint_file = os.path.join(self.args.previous_task_checkpoint, "pytorch_model.bin")
                    elif os.path.exists(os.path.join(self.args.previous_task_checkpoint, "model.safetensors")):
                        # Handle safetensors format
                        from safetensors.torch import load_file
                        prev_state_dict = load_file(os.path.join(self.args.previous_task_checkpoint, "model.safetensors"))
                    
                    if checkpoint_file:
                        checkpoint = torch.load(checkpoint_file, map_location='cpu')
                        if 'model' in checkpoint:
                            prev_state_dict = checkpoint['model']
                        else:
                            prev_state_dict = checkpoint
                else:
                    # It's a file
                    checkpoint = torch.load(self.args.previous_task_checkpoint, map_location='cpu')
                    if 'model' in checkpoint:
                        prev_state_dict = checkpoint['model']
                    else:
                        prev_state_dict = checkpoint
                
                # Load previous parameters
                for name in self.param_names:
                    if name in prev_state_dict:
                        self.previous_params[name] = prev_state_dict[name].to(self.model.device).clone()
                    else:
                        # If parameter not found in checkpoint, use current model parameter
                        logger.info(f"Warning !!!!!: parameter not found in checkpoint, use current model parameter")
                        for n, p in self.model.named_parameters():
                            if n == name:
                                self.previous_params[name] = p.data.clone()
                                break
                                
                logger.info(f"Loaded previous parameters for {len(self.previous_params)} parameters")
                
            except Exception as e:
                logger.warning(f"Failed to load previous task checkpoint: {e}")
                logger.warning("Initializing previous_params with current model parameters!!!!!")
                for name, param in self.model.named_parameters():
                    if param.requires_grad and name in self.param_names:
                        self.previous_params[name] = param.data.clone()
        else:
            logger.warning("No previous task checkpoint specified!!!!!")
            logger.warning("Initializing previous_params with current model parameters!!!!!")
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in self.param_names:
                    self.previous_params[name] = param.data.clone()
    
    def _save_grad(self, name):
        """Create hook to save gradients during training (TRACE-master style)"""
        def hook(grad):
            if grad is not None:
                # Handle NaN values
                grad = torch.nan_to_num(grad, nan=0.0)
                # Store gradient on GPU for efficiency
                self.grads[name] = grad.clone()
            return grad  # Return grad so it can continue in the computation graph
        return hook
    
    def _register_hooks(self):
        """Register hooks to capture gradients (TRACE-master style)"""
        if self.hooks_registered:
            logger.info("Hooks already registered, skipping...")
            return
            
        logger.info("Registering gradient hooks for EWC...")
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.param_names:
                param.register_hook(self._save_grad(name))
        
        self.hooks_registered = True
        logger.info(f"Registered hooks for {len(self.param_names)} parameters")
    
    def _update_fisher(self):
        """
        Update Fisher information during training (TRACE-master style)
        This is called after each training step to accumulate Fisher information
        """
        if self.train_dataset_length == 0:
            return
            
        for name in self.param_names:
            if name in self.grads:
                grad_norm = self.grads[name].norm().item()
                # if name == 'model.layers.23.self_attn.k_norm.weight':
                #     print(f"Step {self.state.global_step}, {name}: grad_norm = {grad_norm:.6f}")
                # Accumulate: Fisher += (grad^2) / train_length
                # Following TRACE-master's implementation (compute directly on GPU)
                self.fisher[name] += (self.grads[name] ** 2) / self.train_dataset_length
    
    def _ewc_penalty(self):
        """
        Compute EWC regularization penalty (TRACE-master style)
        penalty = sum(Fisher * (current_param - previous_param)^2)
        """
        penalty = 0.0
        aa = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.param_names:
                if name in self.previous_params and name in self.fisher:
                    # EWC penalty: F_i * (θ_i - θ*_i)²
                    # All tensors are already on the same device (GPU)
                    penalty += torch.sum(self.fisher[name] * (param - self.previous_params[name]) ** 2)
                    # print(self.fisher[name])
                    # print(param)
                    # print(self.previous_params[name])
                    aa += 1
        # print('======')
        # print(aa)
        # print(self.param_names)
        return penalty
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Override compute_loss to add EWC regularization (TRACE-master style)
        """
        # Initialize EWC if not done yet
        if not self.ewc_initialized:
            self._initialize_ewc()
            self.ewc_initialized = True
            
        # Compute standard loss
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch)
        
        # Add EWC penalty for tasks after the first one (TRACE-master: if task_num != 0)
        if self.args.task_id > 0 and len(self.previous_params) > 0:
            try:
                ewc_penalty = self._ewc_penalty()
                # TRACE-master uses: loss += 0.5 * lambda_ewc * penalty
                loss = loss + 0.5 * self.args.ewc_lambda * ewc_penalty
                
                # Log EWC loss periodically
                if hasattr(self, 'state') and self.state.global_step % self.args.logging_steps == 0:
                    logger.info(f"Step {self.state.global_step}: EWC penalty = {ewc_penalty.item():.6f}, Total loss = {loss.item():.6f}")
            except Exception as e:
                logger.warning(f"EWC penalty calculation failed: {e}")
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Override training_step to update Fisher information after each step (TRACE-master style)
        """
        # Perform standard training step
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        
        # Update Fisher information after backward pass (TRACE-master style)
        # The gradients have been captured by the hooks during backward()
        self._update_fisher()
        
        return loss
    
    def train(self, *args, **kwargs):
        """
        Override train method to register hooks and save task info after training
        """
        # Initialize EWC if not done yet
        if not self.ewc_initialized:
            self._initialize_ewc()
            self.ewc_initialized = True
        
        # Calculate training dataset length for Fisher normalization
        try:
            train_dataloader = self.get_train_dataloader()
            self.train_dataset_length = len(train_dataloader)
            logger.info(f"Training dataset length: {self.train_dataset_length} batches")
        except Exception as e:
            logger.warning(f"Failed to get train dataloader length: {e}")
            self.train_dataset_length = 1000  # fallback value
        
        # Register hooks before training starts (TRACE-master: retain_grad() at the beginning)
        self._register_hooks()
        
        # Standard training
        logger.info(f"Starting EWC training for task {self.args.task_id}...")
        result = super().train(*args, **kwargs)
        
        # After training, save Fisher information and current parameters for next task
        try:
            self._save_task_info()
        except Exception as e:
            logger.warning(f"Failed to save task info: {e}")
            import traceback
            logger.warning(traceback.format_exc())
        
        return result.self.param_names
    
    def _save_task_info(self):
        """
        Save current task information for next task (TRACE-master style)
        Saves Fisher information and current parameters
        """
        logger.info(f"Saving task information for EWC (Task {self.args.task_id})...")
        
                # Save current parameters as previous parameters for next task
        # TRACE-master: _update_previous_params() after each task
        current_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.param_names:
                current_params[name] = param.data.cpu().clone()  # Only move to CPU when saving
                
        # Save to checkpoint directory
        if self.args.output_dir:
            # Save Fisher information (move to CPU only for saving)
            fisher_save_path = os.path.join(self.args.output_dir, f"fisher_task_{self.args.task_id}.pt")
            fisher_cpu = {name: fisher_tensor.cpu() for name, fisher_tensor in self.fisher.items()}
            torch.save(fisher_cpu, fisher_save_path)
            logger.info(f"Fisher information saved to: {fisher_save_path}")
            
            # Save current parameters
            params_save_path = os.path.join(self.args.output_dir, f"params_task_{self.args.task_id}.pt")
            torch.save(current_params, params_save_path)
            logger.info(f"Parameters saved to: {params_save_path}")
            
            # Save task metadata
            metadata = {
                "task_id": self.args.task_id,
                "ewc_lambda": self.args.ewc_lambda,
                "fisher_sample_size": self.train_dataset_length,
                "num_parameters": len(self.param_names)
            }
            metadata_path = os.path.join(self.args.output_dir, f"ewc_metadata_task_{self.args.task_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"EWC task info saved successfully for task {self.args.task_id}")
            logger.info(f"Next task should use:")
            logger.info(f"  --previous_task_checkpoint {self.args.output_dir}")
            logger.info(f"  --fisher_save_path {fisher_save_path}") 