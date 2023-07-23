import os
import json
import pandas as pd
from typing import Optional
from datasets import load_dataset


class FalconFineTuningDataset:
    def __init__(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature: str, label: str
    ) -> None:
        self.train_df = train_df
        self.test_df = test_df
        self.feature = feature
        self.label = label

    def export_json(self, dataset, name):
        with open(name, "w") as f:
            json.dump(dataset, f)

    def format_text(self, row, format_style: int, test: Optional[bool] = None) -> str:
        """It provides different standard format of texts which were used in the LLM pretraining / fine tuning.
        Format 1:
            ```json
            {
                'text': '### Human: xxxxxxxx ### Assistant'
            }
            ```

        Format 2:
            ```json
            {
                'text' : 'xxxx',
            }
            ```

        Format 3:
            ```json
            {
                'instruction' : 'xxxx',
                'text' : 'xxxx',
                'label': 'xxxx'
            }
            ```
        Note: Here we are only providing the text format for 1,2. Because format 3 requires a
        json data structure to build.
        """
        assert format_style == 1 or format_style == 2, "Format options are only 1 or 2"

        feature_row = row[self.feature]
        label_row = row[self.label]

        if not test:
            format1_text = f"### Human: This consumer complaint: {feature_row} is classified into category: ### Assistant: {label_row}"

            format2_text = (
                f"This consumer complaint: {feature_row} is classified into category: {label_row}"
            )
        else:
            format1_text = (
                f"### Human: This consumer complaint: {feature_row} is classified into category:"
            )

            format2_text = f"This consumer complaint: {feature_row} is classified into category:"
        return format1_text if format_style == 1 else format2_text

    def instruction_json_blob(self, row, instruction_prompt):
        """This helps to create the format 3 type of prompt i.e. instruction augmented prompt"""
        instruction_augmented_blob = {
            "instruction": instruction_prompt,
            "input": f"text: {row[self.feature]} is classified into category: ",
            "output": row[self.label],
        }
        return instruction_augmented_blob

    def create_dataset_with_formats(
        self,
        format_style: int,
        instruction_prompt: Optional[str] = None,
        test: Optional[bool] = None,
    ) -> dict:
        """
        Here in this dataset format there will be no seperate sections of instruction or label
        rather will be some format. Hence there are three different formats:

        1. ### Human: This consumer complaint {feature} is classified as: ### Assistant: {label}
        2. This consumer complaint {feature} is classified as: {label}
        3. instruction: <instruction>, input: <input>, output: <output>

        by just mentioning the format we can get the corresponding format json object that can be
        used to train the model.
        """
        assert format_style in [1, 2, 3], "Format style can not be < 1 or > 3"
        dataset_json_object_list = []

        if format_style in [1, 2]:
            if test:
                self.test_df["text"] = self.test_df.apply(
                    self.format_text, format_style=format_style, test=True, axis=1
                )
                for _, row in self.test_df.iterrows():
                    dataset_json_object_list.append({"text": row["text"], "label": row[self.label]})
                return dataset_json_object_list
            else:
                self.train_df["text"] = self.train_df.apply(
                    self.format_text, format_style=format_style, test=False, axis=1
                )
                return self.train_df["text"].apply(lambda text: {"text": text}).tolist()

        for _, row in self.train_df.iterrows() if not test else self.test_df.iterrows():
            dataset_json_object_list.append(self.instruction_json_blob(row, instruction_prompt))
        return dataset_json_object_list

    def get_hf_dataset(
        self,
        format_style: int,
        json_folder_path: str,
        validation_size: int,
        instruction_prompt: Optional[str] = None,
        **kwargs,
    ):
        """
        This will directly provide all the dataset required to fine tuning and testing on the model with
        the required hugging face format.

        Args:
            format_style: The format style (1, 2, 3) for generating the prompt
            json_folder_path: the folder where the json needs to be saved
            validation_size: The size required to do validation during the time of training
            instruction_prompt: Only used if format 3 is been choosen
            shuffle: (bool) If True will be shuffling the dataset randomly
            seed: (int) This will set the seed before doing the randomization to do a reproducibility

        Returns:
            A json with the following keys:
            :key metadata
                :key format_style: The format style which was choosen initially
                :key saved_path: the path where all the json of that format where saved
                :key validation_size: the validation size used
                :key seed: The seed set for doing randomization
                :key shuffle: Whether the dataset was shuffled or not
            :key train_dataset: The dataset (HF) format for training
            :key eval_dataset: The dataset (HF) format for validation
            :key test_dataset: The dataset (HF) format for testing
        """
        train_eval_dataset = self.create_dataset_with_formats(
            format_style=format_style, instruction_prompt=instruction_prompt, test=False
        )

        test_dataset = self.create_dataset_with_formats(
            format_style=format_style, instruction_prompt=instruction_prompt, test=True
        )

        print(f"=> Saving datasets inside: {json_folder_path}/format_{format_style}")
        path_to_save = os.path.join(json_folder_path, f"format_{format_style}")

        if not os.path.exists(path_to_save):
            os.mkdir(path_to_save)

        self.export_json(train_eval_dataset, os.path.join(path_to_save, "train_eval.json"))
        self.export_json(test_dataset, os.path.join(path_to_save, "test.json"))

        # import those with hugging face formats

        train_eval_hf_data = load_dataset(
            "json", data_files=os.path.join(path_to_save, "train_eval.json")
        )
        test_hf_data = load_dataset("json", data_files=os.path.join(path_to_save, "test.json"))

        train_eval_hf_data = train_eval_hf_data["train"].train_test_split(
            test_size=validation_size, **kwargs
        )

        data = {
            "metadata": {
                "format_stye": format_style,
                "train_eval_json_path": os.path.join(path_to_save, "train_eval.json"),
                "test_json_path": os.path.join(path_to_save, "test.json"),
                **kwargs,
            },
            "train_dataset": train_eval_hf_data["train"],
            "eval_dataset": train_eval_hf_data["test"],
            "test_dataset": test_hf_data["train"],
        }
        return data