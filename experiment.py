import os
import torch 
import wandb 
import pandas as pd 
from tqdm.auto import tqdm 
from typing import Optional 
from src.dataset import FineTuningDataset
from src.finetune import FineTuner
from src.config import CONFIG 
from peft import PeftModel

class Experiment:
    def __init__(self, experiment_name: str, config: Optional[dict]=None) -> None:
        """
        Argument:
            :param experiment_name: (str) The Name of the experiment. Make sure the format 
        of the experiment_name is in the form of <user-name>/<experiment-name>.
            :param config: (dict) This is a dictionary which should contain all the configs  

        Note: Try to make unambiguous and descriptive experiment name. 
        """ 
        assert len(experiment_name.split('/')) == 2, \
        'experiment_name should be in the format of <user-name>/<experiment-name>'
        
        self.experiment_name = experiment_name
        self.config = config if config is not None else CONFIG
        
        # make a folder under /home/{instance-user}/{exp-user}/{experiment_name}
        
        self.root_path = f'/home/{os.getlogin()}/LLMHypothesisTesting/experiments'
        if not os.path.exists(self.root_path):
            os.makedirs(self.root_path, exist_ok=True)

        self.exp_user, self.exp_name = experiment_name.split('/')
        self.exp_folder = os.path.join(self.root_path, self.exp_user, self.exp_name)
        
        if not os.path.exists(self.exp_folder):
            os.makedirs(self.exp_folder, exist_ok=True)   
        self.run = wandb.init()
    
    def load_datasets(self, train_csv_path: str, test_csv_path: str, validation_split: float=0.1, instruction_prompt: Optional[str]=None) -> dict:
        """
        Loads a HuggingFace datasets for both train and test datasets. 
        Arguments:
            :param train_csv_path: (str) The csv path for train data
            :param test_csv_path: (str) The csv path for test data
        
        Note: The csv should strictly have the following columns: 'id', 'complaint', 'issue_code'
        """
        dataset_maker = FineTuningDataset(
            train_df = pd.read_csv(train_csv_path), test_df=pd.read_csv(test_csv_path),
            feature = 'complaint', label='issue_code'
        )

        json_folder_path = os.path.join(self.exp_folder, 'hf_data')
        hf_dataset = dataset_maker.get_hf_dataset(
            json_folder_path=json_folder_path,
            validation_size=validation_split,
            instruction_prompt=instruction_prompt
        )
        return hf_dataset

    def finetune(self, instruction_prompt: Optional[str]=None):
        dataset_config = self.config['dataset']
        hf_dataset = self.load_datasets(
            train_csv_path=dataset_config['train_dataset_csv'],
            test_csv_path=dataset_config['test_dataset_csv'],
            instruction_prompt=instruction_prompt
        )
        hf_dataset_config = {
            'train_dataset': hf_dataset['train_dataset'],
            'eval_dataset': hf_dataset['eval_dataset'],
            'dataset_text_field':'text',
            'max_seq_length': dataset_config['max_seq_length']
        }

        output_dir = os.path.join(self.exp_folder, 'model_weights')

        self.config['training_arguments']['output_dir'] = output_dir
        self.config['training_arguments']['run_name'] = f"{self.experiment_name}_{self.config['base_model']['model']}_{self.config['training_arguments']['run_name']}"

        trl_trainer = FineTuner(config=self.config).load_trl_trainer(hf_dataset_config=hf_dataset_config)
        trl_trainer.train()
        #trl_trainer.evaluate()
        wandb.finish()
        print("=> Trained the model successfully")
        del trl_trainer
        torch.cuda.synchronize()


    def evaluate(self, instruction_prompt: str, checkpoint_id: str, limit: Optional[int]=None):
        device='cuda' if torch.cuda.is_available() else 'cpu'
        output_dir = os.path.join(self.exp_folder, 'model_weights')
        checkpoint_dir = os.path.join(output_dir, checkpoint_id)
        finetuner = FineTuner(config=self.config)
        
        model, tokenizer = finetuner.model, finetuner.tokenizer
        model = PeftModel.from_pretrained(
            model,
            checkpoint_dir,
            torch_dtype=torch.bfloat16, is_trainable=False
        )
        test_data = pd.read_csv(self.config['dataset']['test_dataset_csv'])
        data = zip(
            list(test_data['complaint'][:limit]),
            list(test_data['issue_code'][:limit])
        )
        tokens_list = []
        probabilities_list=[]
        token_names_list=[]
        texts = []
        labels = [] 
        tokens_list = [] 

        counter=0
        limit = limit if limit is not None else len(test_data)
        for text, label in tqdm(enumerate(data), total=len(test_data['complaint'][:limit])):
            texts.append(text)
            labels.append(label)
            counter=counter+1
            if counter in (500,1000,1500,2000):
                print(counter)
            prompt = instruction_prompt.format(
                complaint=text,
                issue_code=''
            )
            input_ids = tokenizer(prompt, return_tensors="pt",padding="max_length").input_ids.to(device)
            peft_model_outputs = model.generate(
                input_ids=input_ids,
                **self.config['evaluation']
            )

            transition_scores = model.compute_transition_scores(
                peft_model_outputs.sequences, 
                peft_model_outputs.scores, 
                normalize_logits=True 
            )

            probabilities = torch.nn.functional.softmax(transition_scores, dim=-1)
            sequences = peft_model_outputs['sequences']
            tokens_list.append(sequences[0].tolist())
            probabilities_list.append(probabilities[0].tolist())
            
            model_inputs = tokenizer.encode(prompt, return_tensors='pt')
            output = model.generate(
                input_ids=model_inputs.to(device),
                **self.config['evaluation']
    
            )

            token_names_list.append(tokenizer.batch_decode(output['sequences'] , skip_special_tokens=True))
            tokens = output['sequences'].detach().cpu().numpy()
            tokens_list.append(tokens)
        
        data_df = pd.DataFrame({
            'id': list(range(1, limit + 1)),
            'text': texts,
            'label': labels,
            'probabilities': [str(prob) for prob in probabilities_list],
            'tokens list': [str(tk_name_list) for tk_name_list in token_names_list],
        })

        wandb_table = wandb.Table(dataframe=data_df)
        self.run.log({"my_table": wandb_table})
        return data_df
        

if __name__ == '__main__':
    exp = Experiment(experiment_name='anindya/trial-v1-flant5-trial2')
    instruction_prompt = """

Classify a customer complaint into either of the following labels:
- purchase_dispute_unresolved
- purchase_dispute_unauthorized_charged
- payment_process_issues
- fees_dispute
- other_dispute
- account_closure_dispute

You are not allowed to output any other words other than purchase_dispute_unresolved or purchase_dispute_unauthorized_charged 
or payment_process_issues or fees_dispute or other_dispute or account_closure_dispute. And only classify the complaint in just 
one class.

So the following complaint: {complaint} is classified into:

### Assistant: {issue_code}
"""
    exp.finetune(instruction_prompt=instruction_prompt)
    results = exp.evaluate(
        instruction_prompt=instruction_prompt,
        checkpoint_id='checkpoint-100', limit=100
    )