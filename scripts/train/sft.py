import torch
import transformers

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from trl import DataCollatorForCompletionOnlyLM, SFTTrainer
from datasets import load_dataset, concatenate_datasets
import multiprocessing

from dataclasses import dataclass, field
from leap_llm.utils.trainer import (
    ModelArguments,
    TrainingArguments,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class DataArguments:
    data_dir: str = field(default=None, metadata={"help": "path to training data"})
    prior_data_dir: str = field(default=None, metadata={"help": "path to prior data"})
    data_dirs: str = field(default=None, metadata={"help": "path to training data directories"})


def main():
    ###### argument parser ######
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    completion_token = "<|response|>"

    if data_args.data_dir is not None:
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
    elif data_args.data_dirs is not None:
        ###### Load and merge datasets from multiple directories ######
        data_dirs = data_args.data_dirs.split(',')

        train_datasets, eval_datasets = [], []
        for data_dir in data_dirs:
            train_datasets.append(load_dataset("json", data_files=f"{data_dir}/train.json", split="train"))
            eval_datasets.append(load_dataset("json", data_files=f"{data_dir}/test.json", split="train"))
        
        N_train = min(len(dataset) for dataset in train_datasets)
        N_eval = min(len(dataset) for dataset in eval_datasets)

        # Initialize merged datasets by selecting equal amounts from each dataset
        train_dataset = concatenate_datasets([
            dataset.shuffle(seed=42).select(range(N_train)) for dataset in train_datasets
        ]).shuffle(seed=42)

        eval_dataset = concatenate_datasets([
            dataset.shuffle(seed=42).select(range(N_eval)) for dataset in eval_datasets
        ]).shuffle(seed=42)
    else:
        raise ValueError("One of the two must be valid: data_dir or data_dirs.")

    ###### load tokenizer ######
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_id, truncation=True, padding=True
    )
    tokenizer.truncation_side = "left"
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def process(row):
        row["response"][0][
            "content"
        ] = f"{completion_token} {row['response'][0]['content']}"
        row["text"] = tokenizer.apply_chat_template(
            row["prompt"] + row["response"], tokenize=False
        )
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

    ###### trainer ######
    collator = DataCollatorForCompletionOnlyLM(
        completion_token,
        tokenizer=tokenizer,
        mlm=False,
        return_tensors="pt",
        pad_to_multiple_of=8,
    )
        
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        peft_config=peft_config,
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        dataset_text_field="text",
        packing=training_args.packing,
        dataset_kwargs={
            "add_special_tokens": False,  # We template with special tokens
            "append_concat_token": False,  # No need to add additional separator token
        },
    )

    trainer.train()

    ###### cleanup ######
    del model
    del trainer
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
