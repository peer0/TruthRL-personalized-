# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Supervised fine-tuning script for decoder language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path open-r1/Qwen2.5-Math-7B-RoPE-300k \
    --dataset_name open-r1/Mixture-of-Thoughts \
    --dataset_config all \
    --eos_token '<|im_end|>' \
    --learning_rate 4.0e-5 \
    --num_train_epochs 5 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/OpenR1-Distill-7B
"""

import logging
import os
import sys
import random

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format


logger = logging.getLogger(__name__)


def truncate_prompt(
    example,
    tokenizer,
    max_length: int = 8192,
):
    prompt = example["prompt"]
    completion = example["completion"]
    messages = [{"role": "user", "content": prompt[0]['content']}, {"role": "assistant", "content": completion[0]['content']}]
    
    tokenized_messages = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, add_special_tokens=True)

    tokenized_prompt = tokenizer.encode(prompt[0]['content'], add_special_tokens=False)
    tokenized_completion = tokenizer.encode(completion[0]['content'], add_special_tokens=False)

    # print(f'>>> tokenized_prompt: {tokenized_prompt}\n{tokenizer.convert_ids_to_tokens(tokenized_prompt)}')

    # print(f'>>> tokenized_messages: {tokenized_messages}\n{tokenizer.convert_ids_to_tokens(tokenized_messages)}')

    # print(f'>>> tokenized_completion: {tokenized_completion}\n{tokenizer.convert_ids_to_tokens(tokenized_completion)}')

    tokenized_prefix = tokenizer.apply_chat_template([{"role": "user", "content": ""}], tokenize=True, add_generation_prompt=False, add_special_tokens=False)
    # print(f'>>> tokenized_prefix: {tokenized_prefix}\n{tokenizer.convert_ids_to_tokens(tokenized_prefix)}')

    prefix_len = len(tokenized_prefix) - 1
    
    # print(f'len of tokenized_messages: {len(tokenized_messages)}')
    # print(f'len of tokenized_completion: {len(tokenized_completion)}')
    # print(f'len of tokenized_prompt: {len(tokenized_prompt)}')
    # print(f'len of system_prompt: {len(tokenized_prefix)}\n\n')

    if len(tokenized_messages) > max_length:
        # only truncate the prompt
        tokenized_prompt = tokenized_prompt[:max_length - len(tokenized_completion) - prefix_len - 6]
        # print(f'>>> truncated tokenized_prompt: {tokenized_prompt}\n{tokenizer.convert_ids_to_tokens(tokenized_prompt)}')
        # print(f'len of tokenized_prompt: {len(tokenized_prompt)}')
        # print(f'len of tokenized_completion: {len(tokenized_completion)}')
    
    prompt_str = tokenizer.decode(tokenized_prompt, skip_special_tokens=True)

    # new_tokenized_messages = tokenizer.apply_chat_template(new_messages, tokenize=True, add_generation_prompt=False)
    # print(f'\n\n>>> new_tokenized_messages: {new_tokenized_messages}\n\n{tokenizer.convert_ids_to_tokens(new_tokenized_messages)}')
    # print(f'len of new_tokenized_messages: {len(new_tokenized_messages)}')
    # print(f'len of tokenized_completion: {len(tokenized_completion)}')
    # print(f'len of tokenized_prompt: {len(tokenized_prompt)}')


    example["prompt"] = [{"content": prompt_str, "role": "user"}]

    return example

def main(script_args, training_args, model_args):
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, tokenizer, and model #
    ######################################
    dataset = get_dataset(script_args)
    tokenizer = get_tokenizer(model_args, training_args)
    model = get_model(model_args, training_args)

    dataset = dataset.map(
        truncate_prompt,
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_length": training_args.max_length,
        },
        num_proc=training_args.dataset_num_proc,
        desc="Truncating prompt",
    )


    if tokenizer.chat_template is None:
        logger.info("No chat template provided, defaulting to ChatML.")
        model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(dataset[script_args.dataset_train_split])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{dataset[script_args.dataset_train_split][index]}")


    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval and training_args.eval_strategy != "no":
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
