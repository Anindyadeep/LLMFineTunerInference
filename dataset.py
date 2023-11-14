import json
import os
from typing import Optional, Generator, Union, List

import pandas as pd
from datasets import load_dataset


class FineTuningDataset:
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature: str, label: str) -> None:
        """FineTuning dataset class for quicky making a Huggingface compatible dataset.
        Arguments:
            :param train_df (pd.DataFrame) The dataframe that will be used for training
            :param test_df  (pd.DataFrame) The dataframe that will be used for testing
            :param feature  (str) The column name that will be used for feature or context for the LLM. 
            
        """
        self.train_df, self.test_df = train_df, test_df 
        self.feature, self.label = feature, label

    def export_json(self, dataset, name):
        with open(name, "w") as f:
            json.dump(dataset, f)


    def format_text(self, row: Generator, instruction_prompt: str, test: Optional[bool]=False) -> Union[str, List[str]]:
        """Builds the instruction prompt.
        """
        feature_row, label_row = row[self.feature], row[self.label] 
        kwargs = {self.feature: feature_row, self.label: '' if test else label_row}
        return instruction_prompt.format(**kwargs)
        
    def build_json_blob(self, row: Generator, instruction_prompt: Optional[str]=None, test: Optional[bool]=False) -> dict:
        """Function to format the text to be used for a following format. 
        Arguments:
            :param row (Generator) the pandas row generator object for fetching the context and labels. 
            :param instruction_prompt (Optional[str]) The instruction prompt that is to be used. 
            :param test (Optional[bool]) Whether to use for testing or not. 
        
        Return:
            A dictionary that will format the text based on the format style and instruction_prompt passed. 
        
        Note: If instruction_prompt is set to None, then it will return a completion set. If instruction_prompt is not None,
        Make sure when creating the instruction prompt, the arguments passed in {} should be same as passed in 
        feature and label, when initialized the class.  
        """
        if instruction_prompt is not None:
            return {
                'text': self.format_text(row, instruction_prompt=instruction_prompt, test=test),
                'label': row[self.label]
            }
        return {
            'text': row[self.feature],
            'label': row[self.label] 
        }

    def create_dataset_with_formats(self, instruction_prompt: Optional[str] = None, test: Optional[bool] = None) -> dict:
        """Builds the whole dataset in either Instruction format or completion format. 

        Arguments:
            :param instruction_prompt (Optional[str]) The instruction prompt that is to be used. 
            :param test (Optional[bool]) Whether to use for testing or not. 
        """
        dataset_json_object_list = []

        for _, row in self.train_df.iterrows() if not test else self.test_df.iterrows():
            dataset_json_object_list.append(
                self.build_json_blob(row, instruction_prompt, test=test)
            )
        return dataset_json_object_list

    def get_hf_dataset(self, json_folder_path: str, validation_size: int, instruction_prompt: Optional[str] = None, **kwargs):
        """Generates a huggingface compatile dataset from the provided train and test dataframe. 

        Args:
            json_folder_path: the folder where the json needs to be saved. 
            validation_size: The size required to do validation during the time of training
            instruction_prompt: Only used if format 3 is been choosen

        Returns:
            A json with the following keys:
            :key train_dataset: The dataset (HF) format for training
            :key eval_dataset: The dataset (HF) format for validation
            :key test_dataset: The dataset (HF) format for testing
        """

        train_eval_dataset = self.create_dataset_with_formats(instruction_prompt=instruction_prompt, test=False)
        test_dataset = self.create_dataset_with_formats(instruction_prompt=instruction_prompt, test=True)

        print(f"=> Saving datasets inside: {json_folder_path}")

        if not os.path.exists(json_folder_path):
            os.makedirs(json_folder_path, exist_ok=True)

        self.export_json(train_eval_dataset, os.path.join(json_folder_path, "train_eval.json"))
        self.export_json(test_dataset, os.path.join(json_folder_path, "test.json"))

        # import those with hugging face formats

        train_eval_hf_data = load_dataset(
            "json", data_files=os.path.join(json_folder_path, "train_eval.json")
        )
        test_hf_data = load_dataset("json", data_files=os.path.join(json_folder_path, "test.json"))

        train_eval_hf_data = train_eval_hf_data["train"].train_test_split(
            test_size=validation_size, **kwargs
        )

        hf_dataset = {
            "train_dataset": train_eval_hf_data["train"],
            "eval_dataset": train_eval_hf_data["test"],
            "test_dataset": test_hf_data["train"],
        }
        return hf_dataset

