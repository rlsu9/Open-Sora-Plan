#!/bin/bash
MODEL_DIR=./reliance/Open-Sora-Plan-v1.2.0/93x720p
MODEL_DIR2=./reliance/Open-Sora-Plan-v1.2.0/93x720p
OUTPUT_DIR=./93x_without_scale_cm_distill_1e-8


accelerate launch \
  --config_file ./scripts/accelerate_configs/deepspeed_zero2_config.yaml \
   osp_lmcm_x.py \
  --pretrained_teacher_model=$MODEL_DIR \
  --pretrained_student_model=$MODEL_DIR2 \
  --image_dir="./image_output" \
  --output_dir=$OUTPUT_DIR \
  --cache_dir="./cache" \
  --num_train_inferences=8 \
  --lr_scheduler="constant" \
  --mixed_precision=fp16 \
  --resolution=1024 \
  --learning_rate=1e-8 \
  --train_batch_size=1 \
  --max_train_samples=583747 \
  --max_train_steps=50000 \
  --dataloader_num_workers=4 \
  --train_shards_path_or_url='./reliance/video_mixkit_65f_54735.jsonl' \
  --checkpointing_steps=50 \
  --checkpoints_total_limit=100 \
  --gradient_accumulation_steps=1 \
  --use_8bit_adam \
  --report_to=wandb \
  --resume_from_checkpoint "latest" \
  --text_encoder_name=google/mt5-xxl