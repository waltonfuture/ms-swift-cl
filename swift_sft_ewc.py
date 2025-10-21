#!/usr/bin/env python3
"""
EWC-enabled SFT command for ms-swift.
This allows using EWC trainer through command line.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

if int(os.environ.get('UNSLOTH_PATCH_TRL', '0')) != 0:
    import unsloth

from swift.llm.train.sft import SwiftSft
from swift.llm.argument import TrainArguments
from swift.trainers import EWCTrainer, EWCTrainingArguments
from swift.trainers.trainer_factory import TrainerFactory

class EWCSft(SwiftSft):
    """EWC-enabled SFT class"""
    
    def run(self):
        args = self.args
        
        # If EWC is requested, override the task_type
        if hasattr(args, 'use_ewc') and args.use_ewc:
            args.task_type = 'ewc'
        
        return super().run()

def sft_main_ewc(args=None):
    """Main function for EWC-enabled SFT"""
    return EWCSft(args).main()

if __name__ == '__main__':
    sft_main_ewc()
