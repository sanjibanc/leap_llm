import torch
import transformers

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DPOTrainer
from datasets import load_dataset, concatenate_datasets
import multiprocessing

from dataclasses import dataclass, field
from leap_llm.utils.trainer import (
    ModelArguments,
    DPOTrainingArguments,
    get_peft_config,
    get_quantization_config,
)
from peft import PeftModel

@dataclass
class DataArguments:
    data_dir: str = field(default=None, metadata={"help": "path to training data"})
    prior_data_dir: str = field(default=None, metadata={"help": "path to prior data"})

def main():
    ###### argument parser ######
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, DPOTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    ###### load dataset ######
    train_dataset = load_dataset(
        "json", data_files=f"{data_args.data_dir}/train.json", split="train"
    ).shuffle(seed=42)

    eval_dataset = load_dataset(
        "json", data_files=f"{data_args.data_dir}/test.json", split="train"
    ).shuffle(seed=42)

    if data_args.prior_data_dir is not None:
        prior_train_dataset = load_dataset(
            "json", data_files=f"{data_args.prior_data_dir}/train.json", split="train"
        ).shuffle(seed=42)

        if len(prior_train_dataset) > len(train_dataset):
            prior_train_dataset = prior_train_dataset.select(range(len(train_dataset)))

        train_dataset = concatenate_datasets([train_dataset, prior_train_dataset])
        train_dataset = train_dataset.shuffle(seed=42)

    ###### load tokenizer ######
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_id, truncation=True, padding=True
    )
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def process(row):
        row["prompt"] = tokenizer.apply_chat_template(row["prompt"], tokenize=False)
        row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
        row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
        return row

    train_dataset = train_dataset.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )
    eval_dataset = eval_dataset.map(
        process,
        num_proc=multiprocessing.cpu_count(),
        load_from_cache_file=False,
    )

    ###### load model ######
    quantization_config = get_quantization_config(model_args)
    peft_config = get_peft_config(model_args)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_id,
        device_map="auto",
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        quantization_config=quantization_config,
    )
    
    ref_model = None
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_id,
            device_map="auto",
            attn_implementation=model_args.attn_implementation,
            torch_dtype=model_args.torch_dtype,
            quantization_config=quantization_config,
        )
        
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    
    trainer.train()

    ###### cleanup ######
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
