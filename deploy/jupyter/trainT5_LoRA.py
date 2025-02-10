################################################################################
# Copyright (C) 2025 Benjamin Marsh, Shaun Monera
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# This project utilizes modified code from: https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/tuning/distilling_step_by_step/distilling_step_by_step.ipynb
# Changes to the code include, but are not limited to: Support for PEFT methods, support for custon datasets, support for distributed computing methods,
# Support for multiple teacher models, support for FLAN-T5 models, support for accelerate, support for full hyperparameter search
#
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
################################################################################


# Imports

from tempfile import TemporaryDirectory

import evaluate
import torch
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    set_seed,
)
from typing import Tuple, Callable
from transformers import AutoTokenizer
import ray
import ray.train
from ray.train import Checkpoint, DataConfig, ScalingConfig
from ray.train.torch import TorchTrainer
# Load Libraries
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from datasets import DatasetDict, load_dataset
from typing import Dict, Any, List
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
)
import pandas as pd
import torch
import torch.distributed as dist
from transformers import Seq2SeqTrainingArguments
from transformers.trainer_utils import set_seed
import logging

logger = logging.getLogger("ray.job")
SHARED_STORAGE = "/home/ray/src/fine-tuning/Data"


def train_func(config):
    """training function that launches on each worker."""
    SOURCE_DATASET_PATH = "/home/ray/src/fine-tuning/Code/finetunedataset_v2_prompt_tags_removed.csv" # @param {type:"string"} directory path to DSS dataset
    PRETRAINED_BASE_MODEL = "/home/ray/src/models/models--google--flan-t5-large/snapshots/0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"   # @param {type:"string"} directory path to student model
    MAX_INPUT_LENGTH = 1400  # @param {type:"integer"}
    MAX_OUTPUT_LENGTH = 1024  # @param {type:"integer"}
    # Training Hyperparameters
    RUN_ID = 4  # @param {type:"integer"} run ID to set the seed
    LORA_ALPHA = 512  # @param {type:"integer"} LoRA Alpha Parameter
    LORA_RANK = 128  # @param {type:"integer"} LoRA Rank Parameter
    LORA_PARAMS = f"Rank_{LORA_RANK}_Alpha_{LORA_ALPHA}"  # @param {type:"string"} output for logs of LoRA Rank and Alpha
    CONFIG_DIR = "/home/ray/src/fine-tuning/Data/Fine-Tuning_Result_Logs/t5-large-LoRA-A512R128"  # @param {type:"string"} directory path to hyperparameter sweep output directory
    CKPT_DIR = f"{CONFIG_DIR}/ckpts/{LORA_PARAMS}"  # @param {type:"string"} directory for model checkpoints
    LOG_DIR = f"{CONFIG_DIR}/logs/{RUN_ID}"  # @param {type:"string"} directory for training logs
    NUM_EPOCHS = 100 # @param {type:"integer"}
    LEARNING_RATE = 5e-5 # @param {type:"float"}
    BATCH_SIZE = 1 # @param {type:"integer"}
    ALPHA = 0.5 # @param {type:"float"}
    LR_MODE = "min"  # @param {type:"string"} lr will be reduced when the quantity monitored has stopped decreasing
    LR_FACTOR = 0.1  # @param {type:"float"} Factor by which the learning rate will be reduced. new_lr = lr * factor.
    LR_PATIENCE = 1  # @param {type:"integer"} The number of allowed epochs with no improvement after which the
    # learning rate will be reduced.

    dataset = load_dataset('csv', data_files=SOURCE_DATASET_PATH)
    dataset = dataset.rename_column('prompt', 'input')
    dataset = dataset.rename_column('dsl', 'label')

    logger.info(f"Pretrained Base Model: {PRETRAINED_BASE_MODEL}")
    logger.info(f"Max Input Length: {MAX_INPUT_LENGTH}")
    logger.info(f"Max Output Length: {MAX_OUTPUT_LENGTH}")
    logger.info(f"Checkpoint Directory: {CKPT_DIR}")
    logger.info(f"Logging Directory: {LOG_DIR}")
    logger.info(f"Number of Training Epochs: {NUM_EPOCHS}")
    logger.info(f"Base Learning Rate: {LEARNING_RATE}")
    logger.info(f"Batch Size per Worker: {BATCH_SIZE}")
    logger.info(f"Alpha: {ALPHA}")
    logger.info(f"Learning Rate Reduction Factor: {LR_FACTOR}")
    logger.info(f"Learning Rate Patience: {LEARNING_RATE}")

    # tokenize the dataset

    tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_BASE_MODEL)

    def tokenize_function(examples: Dict[str, List[Any]]):
        # Encode input to generate predictions and rationales
        model_inputs = tokenizer(
            ["predict: " + text for text in examples["input"]],
            max_length=MAX_INPUT_LENGTH,
            truncation=True
        )
        expl_model_inputs = tokenizer(
            ["explain: " + text for text in examples["input"]],
            max_length=MAX_INPUT_LENGTH,
            truncation=True
        )
        model_inputs["expl_input_ids"] = expl_model_inputs["input_ids"]
        model_inputs["expl_attention_mask"] = expl_model_inputs["attention_mask"]

        # Encode target label and target rationale
        label_output_encodings = tokenizer(
            text_target=examples["label"], max_length=MAX_OUTPUT_LENGTH, truncation=True
        )
        rationale_output_encodings = tokenizer(
            text_target=examples["rationale"],
            max_length=MAX_OUTPUT_LENGTH,
            truncation=True
        )
        model_inputs["labels"] = label_output_encodings["input_ids"]
        model_inputs["expl_labels"] = rationale_output_encodings["input_ids"]

        return model_inputs

    tokenized_dataset = dataset.map(
        tokenize_function,
        remove_columns=["input", "rationale", "label"],
        batched=True,
    )

    # prepare the model and load LoRA config

    def print_number_of_trainable_model_parameters(model):
        trainable_model_params = 0
        all_model_params = 0
        for _, param in model.named_parameters():
            all_model_params += param.numel()
            if param.requires_grad:
                trainable_model_params += param.numel()
        return f"trainable model parameters: {trainable_model_params}\nall model parameters: {all_model_params}" \
               f"\npercentage of trainable model parameters: {100 * trainable_model_params / all_model_params:.2f}%"

    model = AutoModelForSeq2SeqLM.from_pretrained(PRETRAINED_BASE_MODEL)

    # load LoRA config

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules='all-linear',
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM  # Task type for FLAN-T5
    )

    model = get_peft_model(model, lora_config)
    print(print_number_of_trainable_model_parameters(model))

    print('loaded model')
    '''
    Since we need to generate predictions for both the answer as well as 
    the rationale on each training and prediction step, we will use a custom 
    DataCollator which will take each batch of features and return two sets of 
    features and labels, one each for the answer and for the rationale
    '''
    class TaskPrefixDataCollator(DataCollatorForSeq2Seq):
        def __call__(self, features, return_tensors=None):
            features_df = pd.DataFrame(features)

            # Generate features for answers
            ans_features = features_df.loc[
                           :, features_df.columns.isin(["labels", "input_ids", "attention_mask"])
                           ].to_dict("records")
            ans_features = super().__call__(ans_features, return_tensors)

            # Generate features for explanations
            expl_features = (
                features_df.loc[
                :,
                features_df.columns.isin(
                    ["expl_labels", "expl_input_ids", "expl_attention_mask"]
                ),
                ]
                .rename(
                    columns={
                        "expl_labels": "labels",
                        "expl_input_ids": "input_ids",
                        "expl_attention_mask": "attention_mask",
                    }
                )
                .to_dict("records")
            )
            expl_features = super().__call__(expl_features, return_tensors)

            return {
                "ans": ans_features,
                "expl": expl_features,
            }

    data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)

    # Prepare trainer for multi-task training

    '''
    Similarly, we will use a custom Trainer for training the model, 
    which takes into account both the losses for answer generation as 
    well as rationale generation. We will use a hyperparameter alpha to 
    control the relative contribution of the two losses to the overall model loss
    '''

    class TaskPrefixTrainer(Seq2SeqTrainer):
        def __init__(self, alpha, output_rationale, **kwargs):
            super().__init__(**kwargs)
            self.alpha = alpha
            self.output_rationale = output_rationale

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            ans_outputs = model(**inputs["ans"])
            expl_outputs = model(**inputs["expl"])

            loss = self.alpha * ans_outputs.loss + (1.0 - self.alpha) * expl_outputs.loss

            return (
                (loss, {"ans": ans_outputs, "expl": expl_outputs})
                if return_outputs
                else loss
            )

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            ans_outputs = super().prediction_step(
                model, inputs["ans"], prediction_loss_only=False, ignore_keys=ignore_keys
            )
            if self.output_rationale:
                expl_outputs = super().prediction_step(
                    model,
                    inputs["expl"],
                    prediction_loss_only=False,
                    ignore_keys=ignore_keys,
                )
            else:
                expl_outputs = ans_outputs  # placeholder only

            loss = self.alpha * ans_outputs[0] + (1 - self.alpha) * expl_outputs[0]

            return (
                loss,
                [ans_outputs[1], expl_outputs[1]],
                [ans_outputs[2], expl_outputs[2]],
            )

        # Train the model

    lr_scheduler_kwargs = {
        "mode": LR_MODE,
        "factor": LR_FACTOR,
        "patience": LR_PATIENCE,

    }

    set_seed(RUN_ID)
    print(' set seed run,', RUN_ID)

    training_args = Seq2SeqTrainingArguments(
        CKPT_DIR,
        remove_unused_columns=False,
        eval_strategy="epoch",  # evaulate based on epochs
        save_strategy="epoch",  # save every epoch
        logging_dir=LOG_DIR,
        logging_strategy="epoch",  # log after evey epoch
        log_level="warning",  # log warnings and error messages
        num_train_epochs=NUM_EPOCHS,  # number of epochs
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        seed=RUN_ID,
        lr_scheduler_type="reduce_lr_on_plateau",
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        predict_with_generate=True,
        generation_max_length=MAX_OUTPUT_LENGTH,
        #    metric_for_best_model='eval_test_loss',
        local_rank=-1,
        bf16=False,  # use 32-bit training, default
        prediction_loss_only=False,
        report_to='all'  # default, reports logging values to all integrations (MLFLow, Tensorboard etc)
    )

    # let's try splitting up the dataset...
    tokenized_dataset = tokenized_dataset['train'].train_test_split(test_size=0.2,seed=42)

    trainer_kwargs = {
        "alpha": ALPHA,
        "output_rationale": False,
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_dataset["train"],
        "eval_dataset": tokenized_dataset["test"],
        "data_collator": data_collator,
        "tokenizer": tokenizer,
    }


    accelerator = Accelerator()

    trainer = accelerator.prepare(TaskPrefixTrainer(**trainer_kwargs))

    trainer.train()

if __name__ == "__main__":
    # fine-tune the target model

    trainer = TorchTrainer(
        train_func,
        scaling_config=ScalingConfig(
            num_workers=8,
            use_gpu=True
        ),
        # If running in a multi-node cluster, this is where you
        # should configure the run's persistent storage that is accessible
        # across all worker nodes.
        # run_config=ray.train.RunConfig(storage_path=SHARED_STORAGE)
        train_loop_config=ray.train.RunConfig(storage_path=SHARED_STORAGE)
    )
    trainer.fit()
