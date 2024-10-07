#!/bin/bash


# iter = 2
# Check if an iteration id is provided
if [ $# -eq 0 ]
then
    echo "No iteration id provided. Usage: $0 <iter_id>"
    exit 1
fi

# Define the parameter pairs to sweep over
# declare -a learning_rates=("1e-6" "5e-7" "1e-6" "5e-7")
# declare -a betas=("0.01" "0.01" "0.1" "0.1")

declare -a learning_rates=("5e-7")
declare -a betas=("0.01")

num_train_epochs=1
save_prefix="save/240902/"

iter_id=$1
prev_iter_id=$((iter_id-1))

echo "Iteration id: $1"

# Loop over the parameter pairs
for i in ${!learning_rates[@]}; do
    learning_rate=${learning_rates[$i]}
    beta=${betas[$i]}
    
    save_model_name="alfworld_dpo_lr${learning_rate}_bt${beta}_ep${num_train_epochs}/iter${iter_id}"
    
    echo "Running training with learning_rate=${learning_rate}, beta=${beta}"
    echo "Saving model as: ${save_model_name}"
    
    python scripts/train/dpo.py \
        --data_dir data/alfworld/pref/iter${iter_id} \
        --output_dir ${save_prefix}/${save_model_name} \
        --model_id leap-llm/Meta-Llama-3-8B-Instruct-dpo-alfworld-lr5e-7-bt0.01-ep1-iter1 \
        --per_device_train_batch_size 2 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 8 \
        --num_train_epochs ${num_train_epochs} \
        --gradient_checkpointing True \
        --max_length 8000 \
        --max_prompt_length 6000 \
        --torch_dtype bfloat16 \
        --optim adamw_torch_fused \
        --learning_rate ${learning_rate} \
        --eval_strategy steps \
        --eval_steps 250 \
        --save_strategy steps \
        --save_steps 250 \
        --save_total_limit 5 \
        --load_best_model_at_end True \
        --metric_for_best_model eval_loss \
        --use_peft True \
        --beta ${beta} \
        --lora_alpha 64 \
        --lora_r 128 \
        --lora_dropout 0.05 \
        --lr_scheduler_type cosine \
        --max_grad_norm 0.3 \
        --warmup_steps 10 \
        --bf16 \
        --seed 42 \
        --report_to wandb \
        --logging_first_step \
        --logging_steps 10 \
        --push_to_hub False
done