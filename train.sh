CUDA_VISISBLE_DEVICES=0 accelerate launch --config_file='configs/accelerator_config.yaml' tangoflux/train.py  --checkpointing_steps="1000" --save_every=5 --config='configs/tangoflux_config.yaml'
