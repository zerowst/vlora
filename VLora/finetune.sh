output_model=lora_model
if [ ! -d ${output_model} ];then  
    mkdir ${output_model}
fi
source ~/anaconda3/bin/activate chatdoctor
export PYTHONPATH=$python:/data1/songtao/llama2_cn

model_path=ziqingyang/chinese-alpaca-2-13b

#cp ./finetune_other.sh ${output_model}
#    --load_in_bits 4 \
#export CUDA_VISIBLE_DEVICES=2

deepspeed --include localhost:1 --master_port 10001 finetune.py \
    --model_name_or_path ${model_path} \
    --train_files data/train_combined.csv \
    --validation_files  data/eval_combined.csv \
    --resume_from_checkpoint True \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --do_train \
    --do_eval \
    --use_fast_tokenizer false \
    --output_dir lora_model \
    --evaluation_strategy  steps \
    --max_eval_samples 100 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --gradient_accumulation_steps 2 \
    --load_in_bits 4 \
    --num_train_epochs 6 \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --target_modules q_proj,k_proj,v_proj,o_proj,down_proj,gate_proj,up_proj \
    --logging_dir ${output_model}/logs \
    --logging_strategy steps \
    --logging_steps 2 \
    --save_strategy steps \
    --preprocessing_num_workers 10 \
    --save_steps 50 \
    --eval_steps 50 \
    --save_total_limit 20 \
    --seed 42 \
    --disable_tqdm false \
    --ddp_find_unused_parameters false \
    --block_size 2048 \
    --report_to wandb \
    --overwrite_output_dir \
    --deepspeed ds_config_zero2.json \
    --ignore_data_skip true \
    --gradient_checkpointing \
    --torch_dtype float32
    --ddp_timeout 18000000 \
    | tee -a ${output_model}/train.log

    
