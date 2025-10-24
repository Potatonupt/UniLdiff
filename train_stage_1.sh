CUDA_VISIBLE_DEVICES=4,5 accelerate launch --config_file='./dataloader/accelerate_config.yaml' --mixed_precision="fp16" ./train_SDXL_stage_1.py --pretrained_model_name_or_path='/data/czh/models/RealVisXL_V4.0' --pretrained_vae_model_name_or_path='/data/czh/models/sdxl-vae-fp16-fix' --vae_precision='fp16'  --resolution=512 --random_flip --train_batch_size=16 --gradient_accumulation_steps=4 --gradient_checkpointing --max_train_steps=6000 --learning_rate=5e-5 --max_grad_norm=1.0 --lr_scheduler="cosine" --lr_warmup_steps=300 --output_dir="train_FaithDiff_stage_1_offline" --seed=42 --allow_tf32 --noise_offset=0.02 --use_ema --validation_steps=5000 --checkpointing_steps=1000 --ema_update_interval=100



