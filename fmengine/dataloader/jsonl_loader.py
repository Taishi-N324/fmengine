import copy
import torch
import deepspeed
import transformers
from dataclasses import dataclass
from datasets import load_dataset
from itertools import chain
from torch.utils.data.dataloader import DataLoader
from typing import Dict, List

from fmengine.utils import logger_rank0 as logger


@dataclass
class AutoregressiveLanguageModelDataCollator(object):
    """
    Collate for autoregressive language models
    """

    tokenizer: transformers.PreTrainedTokenizer
    ignore_index: int = -100

    def get_attn_mask(self, input_ids):
        """
        Get triangular attention mask for a given sequence length / device.
        """
        bs = input_ids.shape[0]
        seq_length = input_ids.shape[1]
        # lower triangular attention mask
        mask = torch.tril(torch.ones((bs, seq_length, seq_length))).view(
            bs, 1, seq_length, seq_length
        )
        # convert to binary
        return mask < 0.5

    def get_position_ids(self, input_ids):
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def __call__(self, samples: List) -> Dict[str, torch.Tensor]:
        input_ids = [sample["input_ids"] for sample in samples]
        labels = copy.deepcopy(input_ids)
        # shifting input_ids & labels
        # https://d2l.ai/chapter_recurrent-neural-networks/language-model.html#learning-language-models
        input_ids = [input_id[:-1] for input_id in input_ids]
        labels = [label[1:] for label in labels]
        input_ids = torch.stack(input_ids)
        labels = torch.stack(labels)
        labels = torch.where(
            labels == self.tokenizer.pad_token_id, self.ignore_index, labels
        )
        return (
            (
                input_ids,
                self.get_position_ids(input_ids),
                self.get_attn_mask(input_ids),
            ),
            labels,
        )


def get_jsonl_dataloader(jsonl_path, tokenizer, args):
    data_collator = AutoregressiveLanguageModelDataCollator(tokenizer)
    ctx_length = args.get("seq_length", 1024) + 1  # +1 for shifting
    streaming = args.get("streaming", False)
    seed = args.get("seed", 42)
    batch_size = args.get("batch_size", 1)

    def tokenize(examples):
        examples = tokenizer(
            examples["text"],
            truncation=True,
            max_length=ctx_length
        )
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        if total_length >= ctx_length:
            total_length = (total_length // ctx_length) * ctx_length
        result = {
            k: [t[i : i + ctx_length] for i in range(0, total_length, ctx_length)]
            for k, t in concatenated_examples.items()
        }
        return result

    raw_datasets = load_dataset(
        "json", split="train", data_files=jsonl_path, streaming=streaming
    ).shuffle(seed=seed)

    raw_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets.column_names
    ).with_format("torch")

<<<<<<< HEAD
    def __iter__(self):
        if self.it is None:
            self.it = self.get_stream()
        return self.it

def get_jsonl_dataloader(
            path_to_jsonl_file: str,
            tokenizer: Tokenizer,
            num_workers = 0,
            state_dict = None,
            streaming = False,
            args = None
        ):
    seed = args.get('seed', 3407)
    seq_length = args.get('seq_length', 1024)
    batch_size = args.get('batch_size', 1)
    data_group_size = args.get('data_group_size', 1)
    shuffle = args.get('shuffle', False)
    data = load_dataset(
            'json',
            split='train',
            data_files=path_to_jsonl_file,
            streaming=streaming
        ).shuffle(seed=seed).with_format('torch')
    
    stream_dataset = JSONLDataset(data, tokenizer, seq_length)
    collator = AutoregressiveLanguageModelDataCollator(tokenizer)
    if state_dict:
        stream_dataset.load_state_dict(state_dict)

    train_data_loader = torch.utils.data.DataLoader(
        stream_dataset,
        batch_size = batch_size * data_group_size,
        shuffle = shuffle,
        num_workers = num_workers,
        pin_memory = False,
        collate_fn = collator
    )
    return iter(deepspeed.utils.RepeatingLoader(train_data_loader))
=======
    dataloader = DataLoader(
        raw_datasets, shuffle=False, collate_fn=data_collator, batch_size=batch_size
    )
    return iter(deepspeed.utils.RepeatingLoader(dataloader))
>>>>>>> 209ab9141711a452173e1587497e50ed07aa63b2
