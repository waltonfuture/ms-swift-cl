#!/bin/bash

CURRENT_MODEL_PATH="/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/models/Qwen3-4B-Instruct-2507"

DATASETS=(
    "/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ContinualRL/data/distilled_data/medmcqa-train--distilled--gpt-oss-20b.json"
    "/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ContinualRL/data/distilled_data/medmcqa-train--distilled--Baichuan-M2-32B.json"
    "/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ContinualRL/data/distilled_data/medmcqa-train--distilled--Seed-OSS-36B-Instruct.json"
    "/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ContinualRL/data/distilled_data/medmcqa-train--distilled--Qwen3-30B-A3B-Instruct-2507.json"
    "/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ContinualRL/data/distilled_data/medmcqa-train--distilled--medgemma-27b-text-it.json"
    "/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ContinualRL/data/distilled_data/medmcqa-train--distilled--gpt-oss-120b.json"
    "/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ContinualRL/data/distilled_data/medmcqa-train--distilled--GLM-4.5-Air.json"
    "/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ContinualRL/data/distilled_data/medmcqa-train--distilled--GLM-4.5-FP8.json"
    "/inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ContinualRL/data/distilled_data/medmcqa-train--distilled--Qwen3-235B-A22B-Instruct-2507-FP8.json"
)


BASE_OUTPUT_DIR="output_continual_learning"

run_index=0


# 遍历DATASETS数组中的每一个数据集路径
for DATASET_PATH in "${DATASETS[@]}"; do
    
    DATASET_BASENAME=$(basename "$DATASET_PATH" .json)
    OUTPUT_DIR="${BASE_OUTPUT_DIR}"

    echo "=========================================================================="
    echo "Starting training run #${run_index} for dataset: ${DATASET_BASENAME}"
    echo "Using model from: ${CURRENT_MODEL_PATH}"
    echo "Output will be saved to: ${OUTPUT_DIR}"
    echo "Expecting run directory pattern: v${run_index}-*"
    echo "=========================================================================="

    CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 swift sft \
      --model "${CURRENT_MODEL_PATH}" \
      --train_type full \
      --dataset "${DATASET_PATH}" \
      --num_train_epochs 1 \
      --gradient_accumulation_steps 8 \
      --learning_rate 1e-5 \
      --torch_dtype bfloat16 \
      --save_total_limit 2 \
      --max_length 16384 \
      --warmup_ratio 0.05 \
      --save_steps 50 \
      --output_dir "${OUTPUT_DIR}" \
      --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
      --deepspeed zero2 \
      --save_only_model true

    
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "Error: Training failed. Output directory '${OUTPUT_DIR}' not found."
        exit 1
    fi

    RUN_DIR_PATTERN="${OUTPUT_DIR}/v${run_index}-*"

    # 检查是否有匹配的目录
    if ! ls -d ${RUN_DIR_PATTERN} > /dev/null 2>&1; then
        echo "Error: Could not find a run directory with pattern '${RUN_DIR_PATTERN}'."
        exit 1
    fi
    
    # 找到最新的一个（通常只有一个，但用ls -td更保险）
    LATEST_RUN_DIR=$(ls -td ${RUN_DIR_PATTERN} | head -n 1)

    CHECKPOINTS=($(ls -d "${LATEST_RUN_DIR}"/checkpoint-*/ | sort -V))

    if [ ${#CHECKPOINTS[@]} -eq 0 ]; then
        echo "Error: No checkpoints found inside '${LATEST_RUN_DIR}'."
        exit 1
    fi

    # 步骤 B: 在该目录中找到步数最高的checkpoint（这部分逻辑不变）
    LATEST_CHECKPOINT=$(ls -d "${LATEST_RUN_DIR}"/checkpoint-*/ | sort -V | tail -n 1)

    if [ -z "$LATEST_CHECKPOINT" ]; then
        echo "Error: Could not find any checkpoint inside '${LATEST_RUN_DIR}'."
        exit 1
    fi

    # 获取绝对路径，避免相对路径问题
    LATEST_CHECKPOINT=$(realpath "${LATEST_CHECKPOINT}")

    echo "--------------------------------------------------------------------------"
    echo "Training run #${run_index} for ${DATASET_BASENAME} completed."
    echo "Found latest checkpoint: ${LATEST_CHECKPOINT}"
    echo "This checkpoint will be used for the next training iteration."
    echo "--------------------------------------------------------------------------"

    CURRENT_MODEL_PATH="${LATEST_CHECKPOINT}"

    # BEST_SCORE=-1
    # BEST_CHECKPOINT=""

    # for CKPT in "${CHECKPOINTS[@]}"; do
    #     echo "Evaluating checkpoint: $CKPT"

    #     # 只取 idx/len(questions) 打印出来的分数
    #     SCORE=$(python /inspire/hdd/project/aiforquantum/zhengkaipeng-240108120123/test/codes/ContinualRL/eval/med-eval/medmcqa.py \
    #         --gpus 1 --model_path "$CKPT" 2>&1 | grep -Eo "0\.[0-9]{4}" | head -n 1)

    #     echo "Score for $CKPT = $SCORE"

    #     if awk -v score="$SCORE" -v best="$BEST_SCORE" 'BEGIN { exit !(score > best) }'; then
    #         BEST_SCORE=$SCORE
    #         BEST_CHECKPOINT=$CKPT
    #     fi
    # done

    # if [ -z "$BEST_CHECKPOINT" ]; then
    #     echo "Error: Could not determine best checkpoint."
    #     exit 1
    # fi

    # BEST_CHECKPOINT=$(realpath "$BEST_CHECKPOINT")

    # echo "--------------------------------------------------------------------------"
    # echo "Training run #${run_index} for ${DATASET_BASENAME} completed."
    # echo "Best checkpoint: ${BEST_CHECKPOINT} (score=$BEST_SCORE)"
    # echo "This checkpoint will be used for the next training iteration."
    # echo "--------------------------------------------------------------------------"
    
    # CURRENT_MODEL_PATH="${BEST_CHECKPOINT}"


    ((run_index++))

done
