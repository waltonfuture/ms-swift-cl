#!/bin/bash
# EWC Training Script for ms-swift (TRACE-master style implementation)
# Usage: ./train_ewc.sh <task_id> [previous_checkpoint] [fisher_path]
#
# Implementation follows TRACE-master approach:
# - Fisher information is accumulated during training (not computed after)
# - Only protects the immediate previous task
# - Fisher is overwritten by each new task

# Default parameters (modify as needed)
TASK_ID=${1:-0}  # Current task ID (0 for first task)
PREVIOUS_CHECKPOINT=${2:-""}  # Previous task checkpoint path
FISHER_PATH=${3:-""}  # Fisher information path from previous task

# Model and data paths
# MODEL_PATH="/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/models/Qwen3-4B-Instruct-2507"
MODEL_PATH="/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ms-swift/output_continual_learning1/v8-20250928-043806/checkpoint-5028"
DATASET_PATH="/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ms-swift/data_test/combined_medical_data--distilled--Qwen3-4B-Instruct-2507.json"
OUTPUT_DIR="output_ewc/task_${TASK_ID}"

# EWC parameters
EWC_LAMBDA=400.0  # EWC regularization strength (same as TRACE-master default)
# Note: fisher_sample_size is no longer used - Fisher is computed over entire training dataset

# Create output directory
mkdir -p $OUTPUT_DIR

# Display task information
echo "========================================"
echo "EWC Training (TRACE-master style)"
echo "========================================"
echo "Task ID: $TASK_ID"
echo "Output directory: $OUTPUT_DIR"
echo "EWC Lambda: $EWC_LAMBDA"
if [ $TASK_ID -eq 0 ]; then
    echo "This is the FIRST task - no EWC penalty will be applied"
    echo "Fisher information will be accumulated during training"
else
    echo "This is task #$TASK_ID - EWC penalty WILL be applied"
    echo "Previous checkpoint: $PREVIOUS_CHECKPOINT"
    echo "Fisher information: $FISHER_PATH"
fi
echo "========================================"
echo ""

# Prepare the command
SWIFT_CMD="CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft"

# Basic training parameters
SWIFT_CMD="$SWIFT_CMD --model $MODEL_PATH"
SWIFT_CMD="$SWIFT_CMD --train_type full"
SWIFT_CMD="$SWIFT_CMD --dataset $DATASET_PATH"
SWIFT_CMD="$SWIFT_CMD --num_train_epochs 2"
SWIFT_CMD="$SWIFT_CMD --gradient_accumulation_steps 8"
SWIFT_CMD="$SWIFT_CMD --torch_dtype bfloat16"
SWIFT_CMD="$SWIFT_CMD --save_total_limit 4"
SWIFT_CMD="$SWIFT_CMD --split_dataset_ratio 0.05"
SWIFT_CMD="$SWIFT_CMD --eval_steps 50"
SWIFT_CMD="$SWIFT_CMD --save_steps 50"
SWIFT_CMD="$SWIFT_CMD --output_dir $OUTPUT_DIR"
SWIFT_CMD="$SWIFT_CMD --gradient_checkpointing_kwargs '{\"use_reentrant\": false}'"
# Note: Do NOT use DeepSpeed with EWC - it may cause incorrect Fisher computation
# SWIFT_CMD="$SWIFT_CMD --deepspeed zero2"
SWIFT_CMD="$SWIFT_CMD --save_only_model true"

# Add EWC parameters
SWIFT_CMD="$SWIFT_CMD --task_type ewc"
SWIFT_CMD="$SWIFT_CMD --ewc_lambda $EWC_LAMBDA"
SWIFT_CMD="$SWIFT_CMD --task_id $TASK_ID"

# Add previous task information if provided (required for task_id > 0)
# Add previous task information if provided (required for task_id > 0)
if [ $TASK_ID -gt 0 ]; then
    # 处理 previous checkpoint
    if [ -z "$PREVIOUS_CHECKPOINT" ]; then
        echo "ERROR: For task_id > 0, previous_task_checkpoint is required!"
        exit 1
    fi

    echo "Using previous task checkpoint: $PREVIOUS_CHECKPOINT"
    SWIFT_CMD="$SWIFT_CMD --previous_task_checkpoint $PREVIOUS_CHECKPOINT"

    # 处理 fisher path
    if [ -z "$FISHER_PATH" ] || [ ! -f "$FISHER_PATH" ]; then
        echo "ERROR: For task_id > 0, fisher_save_path must be a valid file!"
        exit 1
    fi

    echo "Using Fisher information from: $FISHER_PATH"
    SWIFT_CMD="$SWIFT_CMD --fisher_save_path $FISHER_PATH"
fi


# Print command for debugging
echo ""
echo "Executing command:"
echo $SWIFT_CMD
echo ""

# Execute the command
eval $SWIFT_CMD

# Check if training was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "==================================="
    echo "Training completed successfully!"
    echo "==================================="
    echo "Task ID: $TASK_ID"
    echo "Output directory: $OUTPUT_DIR"   
    # Save task completion info
    cat > "$OUTPUT_DIR/task_info.json" << EOF
{
    "task_id": $TASK_ID,
    "model_path": "$MODEL_PATH",
    "dataset_path": "$DATASET_PATH",
    "previous_checkpoint": "$PREVIOUS_CHECKPOINT",
    "fisher_path": "$FISHER_PATH",
    "ewc_lambda": $EWC_LAMBDA,
    "output_dir": "$OUTPUT_DIR",
    "completion_time": "$(date)",
    "status": "completed",
    "implementation": "TRACE-master style"
}
EOF
    
    echo "Task information saved to: $OUTPUT_DIR/task_info.json"
    echo ""
    
    # Provide command for next task
    NEXT_TASK_ID=$((TASK_ID+1))
    echo "========================================="
    echo "To train the NEXT task (Task $NEXT_TASK_ID), use:"
    echo "========================================="
    echo "./train_ewc.sh $NEXT_TASK_ID $OUTPUT_DIR $OUTPUT_DIR/fisher_task_${TASK_ID}.pt"
    echo ""
    echo "Note: Make sure to update DATASET_PATH in the script for the next task!"
    echo ""
    
else
    echo ""
    echo "==================================="
    echo "Training FAILED with exit code $?"
    echo "==================================="
    exit 1
fi 