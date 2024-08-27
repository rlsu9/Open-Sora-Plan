#!/bin/bash
# max_train_steps or num_train_epochs
# 设置环境变量
export NCCL_P2P_DISABLE="1"
export NCCL_IB_DISABLE="1"
export CUDA_LAUNCH_BLOCKING=1
MODEL_DIR=/workspace/rlsu/documents/opensora_debug_checkpoints/Open-Sora-Plan-v1.2.0/29x720p
MODEL_DIR2=/workspace/rlsu/documents/opensora_debug_checkpoints/Open-Sora-Plan-v1.2.0/29x720p
OUTPUT_DIR=/workspace/rlsu/documents/opensora_debug_checkpoints/video_distill_output


CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29506 --num_processes 1 osp_lmcm_x.py \
  --pretrained_teacher_model=$MODEL_DIR \
  --pretrained_student_model=$MODEL_DIR2 \
  --image_dir="/workspace/rlsu/documents/video_result/distill_output" \
  --output_dir=$OUTPUT_DIR \
  --cache_dir="/workspace/rlsu/documents/opensora_debug_checkpoints" \
  --num_train_inferences=8 \
  --lr_scheduler="constant" \
  --mixed_precision=fp16 \
  --resolution=1024 \
  --learning_rate=3e-6 \
  --train_batch_size=1 \
  --max_train_samples=583747 \
  --max_train_steps=50000 \
  --dataloader_num_workers=1 \
  --train_shards_path_or_url='/workspace/rlsu/documents/attn_distill/video_mixkit_65f_54735.jsonl' \
  --checkpointing_steps=50 \
  --checkpoints_total_limit=40 \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --report_to=wandb \
  --seed=45369 \
  --text_encoder_name=google/mt5-xxl