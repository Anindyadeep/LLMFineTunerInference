import os
import torch
import json
import warnings
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, 
    BitsAndBytesConfig, TrainingArguments
)
from trl import SFTTrainer

warnings.filterwarnings('ignore')

class FineTuner:
    def __init__(self, config: dict) -> None:
        """
        A config is a single file, that contains configurations for different aspects 
        of the model training. This is how a config file should look like:
        
        config ={
            base_model: {
                model: google/flant5,
                model_type: seq2seq,
                precision: fp32,
                load_in_4bit: True,
                load_in_8bit: False,
                load_with_bnb: False,
            },
            peft_model: {
                r: 16,
                alpha: 32
            },
            training : {
                learning_rate: 0.001,
                epochs: 12
            },
            dataset: {
                train_csv: the path of train csv dataset,
                test_csv: the path for test csv dataset
            }
        }

        And so, on. You can find an example config inside src directory
        """
        
        self.config = config
        self.base_model_config = config['base_model']
        base_model_kwargs = {
            'token': self.base_model_config['token'],
            'trust_remote_code': self.base_model_config['trust_remote_code'],
            'revision': self.base_model_config['revision']
        }

        precision_dict = {
            'fp32': torch.float32,
            'fp16': torch.float16,
            'bf16': torch.bfloat16
        }

        if self.base_model_config['load_in_8bit']:
            base_model_kwargs['load_in_8bit'] = self.base_model_config['load_in_8bit']
            base_model_kwargs['device_map'] = "auto"

        elif self.base_model_config['load_in_4bit']:
            base_model_kwargs['load_in_4bit'] = self.base_model_config['load_in_4bit']
            base_model_kwargs['device_map'] = "auto"

        else:
            print(f"=> Loading model in precision: {self.base_model_config['precision']}")
            base_model_kwargs['torch_dtype'] = precision_dict[self.base_model_config['precision']] 
            
            
        if self.base_model_config['load_with_bnb']:
            print("=> Loading with BitsAndBytes")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=precision_dict[self.base_model_config['precision']],
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            if self.base_model_config['model_type'] == 'causal':
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_config['model'],
                    quantization_config=bnb_config,
                    **base_model_kwargs
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.base_model_config['model'],
                    quantization_config=bnb_config,
                    **base_model_kwargs
                )
        else:
            if self.base_model_config['model_type'] == 'causal':
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.base_model_config['model'],
                    **base_model_kwargs
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.base_model_config['model'],
                    **base_model_kwargs
                )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model_config['model'],
            trust_remote_code = self.base_model_config['trust_remote_code'],
            token=self.base_model_config['token'],
            truncation_side="left", padding_side="right"
        )

        if not self.tokenizer.eos_token:
            if self.tokenizer.bos_token:
                self.tokenizer.eos_token = self.tokenizer.bos_token
                print("bos_token used as eos_token")
            else:
                raise ValueError("No eos_token or bos_token found")
        try:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Some models like CodeGeeX2 have pad_token as a read-only property
        except AttributeError:
            print("Not setting pad_token to eos_token")
            pass
        
        self.peft_config = LoraConfig(
            **config['peft_model']
        )

        print("=> Loaded model and tokenizer successfully !")
        self.training_arguments = TrainingArguments(
            **config['training_arguments']
        )

    
    def load_trl_trainer(self, hf_dataset_config: dict):
        """
        Loads TransformersRL library. We are currently doing SuperVised Fine-tuning through this library.
        Arguments:
            :param hf_dataset_config: (dict) This is a dictionary which should contain train dataset, eval_dataset, max_seq_length and dataset_text_field
        """
        trl_training_args = {
            'model': self.model,
            'peft_config': self.peft_config,
            'tokenizer': self.tokenizer,
            'args': self.training_arguments,
            **hf_dataset_config
        }

        self.model.config.use_cache = False

        print("=> Loaded everything successfully. Ready to finetune !")
        trl_trainer = SFTTrainer(**trl_training_args)
        return trl_trainer 