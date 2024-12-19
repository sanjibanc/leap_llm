#!/bin/bash

# Check if an iteration id is provided
if [ $# -eq 0 ]
then
    echo "No iteration id provided. Usage: $0 <iter_id>"
    exit 1
fi

iter_id=$1
prev_iter_id=$((iter_id-1))

echo "Iteration id: $iter_id"

# Generate a list of data directories from iter0 to iter${iter_id}
data_dirs=""
for ((i=0; i<=iter_id; i++)); do
    data_dirs+="data/alfworld_webshop_intercode/sft/iter${i},"
done

# Remove the trailing comma
data_dirs=${data_dirs%,}

echo "Data directories: $data_dirs"

# Branch based on iteration id
if [ "$iter_id" -eq 0 ]; then
    model_id="meta-llama/Meta-Llama-3-8B-Instruct"
else
    model_id="leap-llm/Meta-Llama-3-8B-Instruct-sft-alfworld-webshop-intercode-iter${prev_iter_id}"
fi

current_date=$(date +"%y%m%d")
echo $current_date


python scripts/train/sft.py \
    --data_dirs "${data_dirs}" \
    --output_dir save/${current_date}/alfworld_webshop_intercode_sft/iter${iter_id} \
    --model_id ${model_id} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --num_train_epochs 1 \
    --gradient_checkpointing True \
    --max_seq_length 6000 \
    --packing False \
    --torch_dtype bfloat16 \
    --optim adamw_torch_fused \
    --learning_rate 3e-5 \
    --evaluation_strategy steps \
    --eval_steps 1000 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 3 \
    --load_best_model_at_end True \
    --metric_for_best_model eval_loss \
    --use_peft True \
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
    --push_to_hub False \