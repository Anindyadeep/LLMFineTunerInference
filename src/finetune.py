import os
import json
import wandb
import warnings

import torch
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig, AutoModelForCausalLM

warnings.filterwarnings('ignore')

# a pipeline something like this 
# fine tune Falcon and save the adapter config some where
# upload it as a weights and bias artifact
# and then get it through langchain
# do not forget to add eos token, padding and truncation where ever required

class SimpleFineTuner:
    def __init__(self, wandb_project_name: str, model_id: str) -> None:
        
        self.run = wandb.init(project=wandb_project_name)
        self.model_id = model_id
    
    def load_base_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        # load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            device_map="auto",
            quantization_config=quantization_config,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code = True)
        return model, tokenizer

    def load_config(self, peft_config_dict: dict, train_args_dict: dict):
        peft_config = LoraConfig(**peft_config_dict)
        training_arguments = TrainingArguments(**train_args_dict)
        return peft_config, training_arguments
    
    def load_trl_trainer(self, model, tokenizer, peft_config_dict: dict, train_args_dict: dict, dataset_config_dict: dict):
        """
        dataset_config_dict must contain the four keys as follows:
        - train_dataset
        - eval_dataset
        - max_seq_length
        - dataset_text_field
        """
        peft_config, training_arguments = self.load_config(peft_config_dict, train_args_dict)
        trl_trainer_args = {
            'model': model,
            'peft_config': peft_config,
            'tokenizer': tokenizer,
            'args': training_arguments,
            **dataset_config_dict
        }

        # disabling model caching to cache configs
        model.config.use_cache = False
        trl_trainer = SFTTrainer(**trl_trainer_args)
        for name, module in trl_trainer.model.named_modules():
            if "norm" in name:
                module = module.to(torch.float32)
        return trl_trainer
    