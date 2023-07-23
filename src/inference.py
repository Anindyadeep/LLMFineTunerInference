import sys
import time
import wandb
import torch
import psutil
import subprocess
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

class SimpleInference:
    def __init__(self, wandb_project_name: str, checkpoint_dir: str) -> None:
        """
        checkpoint_dir: Must in the form of hugging face repo_id/folder 
        example: 'falcon_7b_output/checkpoint-30'
        """
        self.run = wandb.init(project=wandb_project_name)
        self.checkpoint_dir = checkpoint_dir
    
    def get_gpu_usage(self):
        try:
            output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv'])
            gpu_usage = float(output.decode('utf-8').strip().split('\n')[-1].split()[0])
            return gpu_usage
        except subprocess.CalledProcessError:
            print("Error: nvidia-smi not found or GPU not available.")
            return None
        
    def load_finetuned_model(self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

        peft_config = PeftConfig.from_pretrained(self.checkpoint_dir)
        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path,
            return_dict=True,
            device_map='auto',
            trust_remote_code = True,
            quantization_config = quantization_config # may be this can be an arg
        )
        
        model = PeftModel.from_pretrained(model, self.checkpoint_dir)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
        return model, tokenizer
    
    def log_inference_tests(
        self,
        model, 
        tokenizer, 
        test_dataset, 
        max_generation_token, 
        device='cuda:0', 
        table_name=None, 
        using_notebook=True, 
        generation_config=None):
        
        """
        test_dataset: This will be a dataset in HF format
        Track and log these after the table creation in an another table
        i.e. system report
        'RAM usage (bytes)',
        'CPU Usage (%)',
        'GPU Usage (%)',
        'Total Time (seconds)'

        # TODO: Add generation configs too. 
        # TODO: Add other kinds of evaluations too

        # Do not forget to add eos token and padding to left and also truncation and padding
        # still have to research, where and why those are required
        """
        table_name = 'Inference Results Table' if table_name is None else table_name
        if using_notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        
        prediction_report = wandb.Table(columns=[
            'Narrative', 
            'Actual Product', 
            'Predicted Product',
            'RAM usage (bytes)',
            'CPU Usage (%)',
            'GPU Usage (%)',
            'Total Time (seconds)'
        ])

        # load the texts and the labels

        start_time = time.time()
        texts = test_dataset[:]['text']
        labels = test_dataset[:]['label']

        for text, label in tqdm(zip(texts, labels), total=len(texts)):

            # inputs = tokenizer(texts, return_tensors="pt", return_token_type_ids=False, truncation=False, padding=True).to(device)
            # outputs = model.generate(**inputs, max_new_tokens=10)

            inputs = tokenizer(text, return_tensors="pt", return_token_type_ids=False).to(device)
            if generation_config:
                outputs = model.generate(**inputs, max_new_tokens=max_generation_token, generation_config=generation_config)
            else:
                outputs = model.generate(**inputs, max_new_tokens=max_generation_token)
            generated_text = tokenizer.decode(outputs[0]).split("### Assistant:")[1].strip()

            total_time = time.time() - start_time
            final_ram = psutil.virtual_memory().used
            final_cpu = psutil.cpu_percent()
            final_gpu = self.get_gpu_usage()

            prediction_report.add_data(text, label, generated_text, total_time, final_ram, final_cpu, final_gpu)
        self.run.log({table_name: prediction_report})