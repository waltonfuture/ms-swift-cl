#!/bin/bash
# LwF (Learning without Forgetting) Training Script for ms-swift (TRACE-master style implementation)
# Usage: ./train_lwf.sh <task_id> [previous_checkpoint]
#
# Implementation follows TRACE-master approach:
# - Knowledge distillation from previous model's logits on current task data
# - KL divergence loss between current and previous model outputs
# - Only requires previous model checkpoint, no additional information storage

# Default parameters (modify as needed)
TASK_ID=${1:-0}  # Current task ID (0 for first task)
PREVIOUS_CHECKPOINT=${2:-""}  # Previous task checkpoint path

# Model and data paths
# MODEL_PATH="/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/models/Qwen3-4B-Instruct-2507" 
# MODEL_PATH="/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ms-swift/output_continual_learning1/v8-20250928-043806/checkpoint-5028"
MODEL_PATH="/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ms-swift/output_lwf/task_0/v1-20250930-171315/checkpoint-10"
DATASET_PATH="/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ms-swift/data_test/combined_medical_data--distilled--Qwen3-4B-Instruct-2507.json"
OUTPUT_DIR="output_lwf/task_${TASK_ID}"

# LwF parameters
LWF_LAMBDA=1.0  # LwF regularization strength (knowledge distillation weight)
LWF_TEMPERATURE=2.0  # Temperature for knowledge distillation (same as TRACE-master default)

# Create output directory
mkdir -p $OUTPUT_DIR

# Display task information
echo "========================================"
echo "LwF Training (TRACE-master style)"
echo "========================================"
echo "Task ID: $TASK_ID"
echo "Output directory: $OUTPUT_DIR"
echo "LwF Lambda: $LWF_LAMBDA"
echo "LwF Temperature: $LWF_TEMPERATURE"
if [ $TASK_ID -eq 0 ]; then
    echo "This is the FIRST task - no knowledge distillation will be applied"
    echo "Model will be trained normally on the first task"
else
    echo "This is task #$TASK_ID - knowledge distillation WILL be applied"
    echo "Previous checkpoint: $PREVIOUS_CHECKPOINT"
    echo "Previous model logits will be computed for distillation"
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
# LwF is compatible with DeepSpeed (unlike EWC which may have Fisher computation issues)
# SWIFT_CMD="$SWIFT_CMD --deepspeed zero2"  # Uncomment if you want to use DeepSpeed
SWIFT_CMD="$SWIFT_CMD --save_only_model true"

# Add LwF parameters
SWIFT_CMD="$SWIFT_CMD --task_type lwf"
SWIFT_CMD="$SWIFT_CMD --lwf_lambda $LWF_LAMBDA"
SWIFT_CMD="$SWIFT_CMD --lwf_temperature $LWF_TEMPERATURE"
SWIFT_CMD="$SWIFT_CMD --task_id $TASK_ID"

# Add previous task information if provided (required for task_id > 0)
if [ $TASK_ID -gt 0 ]; then
    # Check if previous checkpoint is provided and exists
    if [ -z "$PREVIOUS_CHECKPOINT" ]; then
        echo "ERROR: For task_id > 0, previous_task_checkpoint is required!"
        echo "Usage: ./train_lwf.sh <task_id> <previous_checkpoint>"
        exit 1
    fi

    # Check if previous checkpoint directory exists
    if [ ! -d "$PREVIOUS_CHECKPOINT" ]; then
        echo "ERROR: Previous checkpoint directory does not exist: $PREVIOUS_CHECKPOINT"
        exit 1
    fi


    echo "Using previous task checkpoint: $PREVIOUS_CHECKPOINT"
    SWIFT_CMD="$SWIFT_CMD --previous_task_checkpoint $PREVIOUS_CHECKPOINT"
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
    "lwf_lambda": $LWF_LAMBDA,
    "lwf_temperature": $LWF_TEMPERATURE,
    "output_dir": "$OUTPUT_DIR",
    "completion_time": "$(date)",
    "status": "completed",
    "implementation": "TRACE-master style",
    "method": "LwF (Learning without Forgetting)"
}
EOF
    
    echo "Task information saved to: $OUTPUT_DIR/task_info.json"
    echo ""
    
    # Provide command for next task
    NEXT_TASK_ID=$((TASK_ID+1))
    echo "========================================="
    echo "To train the NEXT task (Task $NEXT_TASK_ID), use:"
    echo "========================================="
    echo "./train_lwf.sh $NEXT_TASK_ID $OUTPUT_DIR"
    echo ""
    echo "Note: Make sure to update DATASET_PATH in the script for the next task!"
    echo "      LwF only needs the previous checkpoint, no additional files required."
    echo ""
    
else
    echo ""
    echo "==================================="
    echo "Training FAILED with exit code $?"
    echo "==================================="
    exit 1
fi 