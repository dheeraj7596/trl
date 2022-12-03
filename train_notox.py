#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
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
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import argparse
import logging
import os
import time
import random
from itertools import chain
import numpy as np
import nltk
from nltk import bigrams

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import transformers
from accelerate import Accelerator, DistributedType
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
    set_seed,
    top_k_top_p_filtering
)
from trl.gpt2 import GPT2HeadWithValueModel
from trl.ppo import PPOTrainer
# from transformers.file_utils import get_full_repo_name
from transformers.utils.versions import require_version

try:
    import wandb

    USE_WANDB = True
except:
    USE_WANDB = False

logger = logging.getLogger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


class LengthSampler:
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))

    def __call__(self):
        return np.random.choice(self.values)


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The configuration name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--analyze_file", type=str, default=None, help="A csv or a json file containing the analysis data."
    )
    parser.add_argument(
        "--validation_split_percentage",
        default=5,
        help="The percentage of the train set used as validation set in case there's no validation split",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--use_slow_tokenizer",
        action="store_true",
        help="If passed, will use a slow tokenizer (not backed by the 🤗 Tokenizers library).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
        choices=MODEL_TYPES,
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help="Optional input sequence length after tokenization. The training dataset will be truncated in block of this size for training. Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=False, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--no_keep_linebreaks", action="store_true", help="Do not keep line breaks when using TXT files."
    )
    parser.add_argument(
        "--analyze_answers", action="store_true", help="Check answers in generated contexts"
    )
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument(
        "--hub_model_id", type=str, help="The name of the repository to keep in sync with the local `output_dir`."
    )
    parser.add_argument("--hub_token", type=str, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default=None,
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_name is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."

    if args.push_to_hub:
        assert args.output_dir is not None, "Need an `output_dir` to create a repo when `--push_to_hub` is passed."

    return args


def main():
    args = parse_args()
    if USE_WANDB:
        accelerator = (
            Accelerator(log_with="wandb", logging_dir=args.output_dir)
        )
        accelerator.init_trackers("NoTox", init_kwargs={"wandb": {"dir": args.output_dir}})
    else:
        accelerator = Accelerator()

    config = {
        "model_name": "gpt2",
        "steps": 20000,
        "batch_size": 128,
        "forward_batch_size": 16,
        "ppo_epochs": 4,
        "txt_in_min_len": 10,
        "txt_in_max_len": 30,
        "txt_out_min_len": 30,
        "txt_out_max_len": 100,
        "lr": 1.41e-5,
        "init_kl_coef": 0.8,
        "target": 6,
        "horizon": 10000,
        "gamma": 1,
        "lam": 0.95,
        "cliprange": .2,
        "cliprange_value": .2,
        "vf_coef": .1,
    }

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                # repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
                repo_name = ""
            else:
                repo_name = args.hub_model_id
            repo = Repository(args.output_dir, clone_from=repo_name)
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
            )
    else:
        data_files = {}
        dataset_args = {}
        if args.train_file is not None:
            data_files["train"] = args.train_file
        if args.validation_file is not None:
            data_files["validation"] = args.validation_file
        extension = args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = not args.no_keep_linebreaks
        raw_datasets = load_dataset(extension, data_files=data_files, **dataset_args)
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                **dataset_args,
            )

    if args.analyze_answers:
        raw_datasets["anal"] = load_dataset('csv', data_files=args.analyze_file)["train"]

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    tokenizer.pad_token = tokenizer.eos_token

    tox_config = AutoConfig.from_pretrained(args.model_name_or_path)

    gpt2_model = GPT2HeadWithValueModel.from_pretrained('gpt2')
    gpt2_model_ref = GPT2HeadWithValueModel.from_pretrained('gpt2')
    gpt2_tox_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=tox_config,
    )

    gpt2_model.resize_token_embeddings(len(tokenizer))
    gpt2_model_ref.resize_token_embeddings(len(tokenizer))
    gpt2_tox_model.resize_token_embeddings(len(tokenizer))

    if USE_WANDB:
        wandb.watch(gpt2_model, log='all')

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = raw_datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    input_size = LengthSampler(config["txt_in_min_len"], config["txt_in_max_len"])
    output_size = LengthSampler(config["txt_out_min_len"], config["txt_out_max_len"])

    def tokenize(sample):
        sample["tokens"] = tokenizer.encode(sample[text_column_name])[:input_size()]
        sample["query"] = tokenizer.decode(sample["tokens"])
        return sample

    with accelerator.main_process_first():
        lm_datasets = raw_datasets.map(
            tokenize,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    gen_kwargs = {
        # "min_length": -1,
        "min_length": config["txt_out_min_len"],
        "top_k": 10,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id
    }

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    # with accelerator.main_process_first():
    #     lm_datasets = tokenized_datasets.map(
    #         group_texts,
    #         batched=True,
    #         num_proc=args.preprocessing_num_workers,
    #         load_from_cache_file=not args.overwrite_cache,
    #         desc=f"Grouping texts in chunks of {block_size}",
    #     )

    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["validation"]

    if args.analyze_answers:
        anal_dataset = lm_datasets["anal"]
        anal_dataloader = DataLoader(
            anal_dataset, collate_fn=default_data_collator, batch_size=1
        )

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    def collater(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=collater, batch_size=config["batch_size"], drop_last=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=collater, batch_size=config["batch_size"], drop_last=True
    )

    gpt2_model, gpt2_model_ref, gpt2_tox_model, train_dataloader, eval_dataloader = accelerator.prepare(
        gpt2_model, gpt2_model_ref, gpt2_tox_model, train_dataloader, eval_dataloader
    )

    ### TRAINING LOOP
    ppo_trainer = PPOTrainer(gpt2_model, gpt2_model_ref, tokenizer, **config)
    total_ppo_epochs = int(np.ceil(config["steps"] / config['batch_size']))

    def toxic_tokenize(texts):
        res = tokenizer(texts, max_length=(config["txt_in_max_len"] + config["txt_out_max_len"]),
                        pad_to_max_length=True,
                        return_tensors="pt").to(accelerator.device)
        res["labels"] = torch.tensor(res["input_ids"]).masked_fill(torch.tensor(res["attention_mask"]) == 0, -100).to(
            accelerator.device)
        return res

    def compute_distinct_score(text):
        tokens = nltk.word_tokenize(text)
        tokens = [token.lower() for token in tokens if len(token) > 1]  # same as unigrams
        unigram_count = len(set(tokens))
        bi_tokens = bigrams(tokens)
        total_bigram_count = len(list(bi_tokens))
        bi_tokens = bigrams(tokens)
        bigram_count = len(set(bi_tokens))
        if total_bigram_count == 0 or len(tokens) == 0:
            return -3
        else:
            return np.mean([unigram_count / len(tokens), bigram_count / total_bigram_count])

    def respond_to_batch(model, queries, txt_len=20, top_k=0, top_p=1.0):
        """Sample text from language model."""
        input_ids = queries
        for i in range(txt_len):
            # Get Logits
            outputs = model(input_ids)
            next_token_logits = outputs[0][:, -1, :]
            next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            # Sample
            probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).squeeze(1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(-1)], dim=-1)
        return input_ids[:, -txt_len:]

    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    for epoch, batch in tqdm(zip(range(total_ppo_epochs), iter(train_dataloader))):
        logs, timing = dict(), dict()
        t0 = time.time()
        query_tensors = [torch.tensor(t).long().to(accelerator.device) for t in batch["tokens"]]

        #### Get response from gpt2
        t = time.time()
        response_tensors = []
        for i in range(config['batch_size']):
            gen_len = output_size()
            response = respond_to_batch(gpt2_model, query_tensors[i].unsqueeze(dim=0), gen_len)
            response_tensors.append(response.squeeze())
            # response = gpt2_model.generate(query_tensors[i].unsqueeze(dim=0),
            #                                max_length=len(query_tensors[i]) + gen_len, **gen_kwargs)
            # response_tensors.append(response.squeeze()[-gen_len:])
        batch['response'] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        timing['time/get_response'] = time.time() - t

        #### Compute Rewards
        t = time.time()
        texts = [q + r for q, r in zip(batch['query'], batch['response'])]
        distinct_scores = np.array([compute_distinct_score(text) for text in texts])
        # distinct_scores = 0
        res = toxic_tokenize(texts)
        batch_size = len(texts)
        with torch.no_grad():
            outputs = gpt2_tox_model(**res)
            lm_logits = outputs.logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = res['labels'][..., 1:].contiguous()
            loss_vec = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1)).detach().cpu().numpy()
            loss_vec = loss_vec.reshape((batch_size, -1))
            loss_vec = loss_vec.sum(axis=-1) / np.count_nonzero(loss_vec, axis=-1)
            # rewards = torch.tensor(np.exp(loss_vec)).to(accelerator.device)
            rewards = torch.tensor(loss_vec) + 0.2 * distinct_scores
            rewards[(rewards < 3) | (rewards > 4)] = -30
            rewards = rewards.to(accelerator.device)
        timing['time/get_toxic_preds'] = time.time() - t

        #### Run PPO step
        t = time.time()
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        timing['time/optimization'] = time.time() - t

        #### Log everything
        timing['time/epoch'] = time.time() - t0
        table_rows = [list(r) for r in zip(batch['query'], batch['response'], rewards.cpu().tolist())]
        logs.update({'game_log': wandb.Table(columns=['query', 'response', 'reward'], rows=table_rows)})
        logs.update(timing)
        logs.update(stats)
        logs['env/reward_mean'] = torch.mean(rewards).cpu().numpy()
        logs['env/reward_std'] = torch.std(rewards).cpu().numpy()
        logs['env/reward_dist'] = rewards.cpu().numpy()
        wandb.log(logs)

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(gpt2_model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


if __name__ == "__main__":
    main()
