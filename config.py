CONFIG = {
    # add train_dataset inside this config later during runtime
    "dataset": {
        'train_dataset_csv': 'GoldenDB/golden_training_db.csv',
        'test_dataset_csv': 'GoldenDB/golden_validation_db.csv',
        'validation_split': 0.1,
        'max_seq_length': 512
    },

    "base_model": {
        "model": "google/flan-t5-base",
        "model_type": "seq2seq",
        "token": False,
        "load_in_4bit": False,
        "load_in_8bit": False,
        "load_with_bnb": True,
        "precision": "fp16",
        "revision": None,
        "trust_remote_code": None
    },

    "peft_model": {
        'r': 64,
        'lora_alpha' : 16,
        'lora_dropout' : 0.1,
        'bias': 'none',
        'task_type': 'SEQ_2_SEQ_LM',
        'target_modules': ["q", "v"],
    }, 

    "training_arguments": {
        "output_dir": "./falcon_7b_output", # change this during the time of experiment
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 4,
        "optim": "adamw_torch",
        "save_steps": 1,
        "save_total_limit":5,
        "logging_steps": 5,
        "learning_rate": 2e-4,
        "max_grad_norm": 0.3,
        "max_steps": 100, # epochs 
        "warmup_ratio": 0.03,
        "lr_scheduler_type": "constant",
        "report_to": "wandb",
        'run_name': "run-flan-t5-base" # same as the above
    },

    'evaluation': {
        'max_new_tokens': 50,
        'num_beams': 1,
        'do_sample': True, 'top_k': 40,
        'top_p': 0.95, 'num_return_sequences':2,
        'output_scores': True,  'return_dict_in_generate': True
    }
}