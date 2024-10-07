import argparse
import yaml
import os
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_and_merge_peft_model(peft_model_id, organization_name, new_model_name, upload_adapters_only=False):
    config_file = os.path.join(peft_model_id, "adapter_config.json")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found in the directory: {peft_model_id}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    base_model_name = config.get('base_model_name_or_path')
    
    if not base_model_name:
        raise ValueError("Base model name not found in the adapter config.")
    
    if not upload_adapters_only:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        model = PeftModel.from_pretrained(base_model, peft_model_id)
        model = model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id, truncation=True, padding=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            peft_model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(peft_model_id, truncation=True, padding=True)
    

    tmp_save_dir = os.path.join("save/tmp/", new_model_name)    
    model.save_pretrained(tmp_save_dir, push_to_hub=True, repo_id=f"{organization_name}/{new_model_name}")
    tokenizer.push_to_hub(repo_id=f"{organization_name}/{new_model_name}")
    
    print(f"Successfully uploaded {new_model_name} to Hugging Face Hub under organization {organization_name}")

parser = argparse.ArgumentParser(description='Upload model')
parser.add_argument('--config', type=str, required=True, help='Path to training config file')
parser.add_argument('--upload_adapters_only', action='store_true', help='Only upload the PEFT adapters without merging the model')
args = parser.parse_args()

with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
config = config['upload_models_to_hf']

model_id_path = config.get('model_id_path', None)
new_model_name = config.get('new_model_name', None)
organization_name = config['organization_name']

def check_and_replace_placeholder(value, placeholder, replacement):
    if placeholder in value:
        if replacement is None:
            raise ValueError(f"{placeholder} placeholder found in the config, but no --iter argument was passed.")
        return value.replace(placeholder, str(replacement))
    return value

peft_model_id = model_id_path

# Ask user for confirmation
confirm = input(f"Do you want to upload {peft_model_id} to {organization_name}/{new_model_name}? [y/n] ").strip().lower()
if confirm != 'y':
    print("Upload cancelled by user.")
    exit()

# Step 8: Load the model and tokenizer
load_and_merge_peft_model(peft_model_id, organization_name, new_model_name, upload_adapters_only=args.upload_adapters_only)