import copy
import torch
import deepspeed
import transformers
from dataclasses import dataclass
from datasets import load_dataset
from itertools import chain
from torch.utils.data.dataloader import DataLoader
from typing import Dict, List
from typing import List, Dict, Any
import numpy as np
from fmengine.utils import logger_rank0 as logger
from streaming import StreamingDataset

@dataclass
class StreamingAutoregressiveLanguageModelDataCollator:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.ignore_index = -100  # usually the value used for ignored indices in PyTorch

    @staticmethod
    def get_attn_mask(input_ids):
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
    @staticmethod
    def get_position_ids(input_ids):
        print("input_ids.shape")
        print(input_ids.shape)
        seq_length = input_ids.shape[1]
        # Position ids.
        position_ids = torch.arange(seq_length, dtype=torch.long)
        return position_ids.unsqueeze(0).expand_as(input_ids)

    def custom_collate(self, samples: List[Dict[str, Any]], max_seq_len: int) -> torch.Tensor:
        tensors = [_read_binary_tokenized_sample(sample, max_seq_len) for sample in samples]
        input_ids = torch.stack(tensors)  # [B, T]

        # Create a tensor of [B, 1]. All values are 1.
        ones = torch.ones((input_ids.size(0), 1), dtype=input_ids.dtype, device=input_ids.device)
        # concatenate ones tensor to the beginning of input_ids
        input_ids = torch.cat((ones, input_ids), dim=1)
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
        position_ids = StreamingAutoregressiveLanguageModelDataCollator.get_position_ids(input_ids)
        attention_mask = StreamingAutoregressiveLanguageModelDataCollator.get_attn_mask(input_ids)
        
        return ((input_ids, position_ids, attention_mask), labels)

def _read_binary_tokenized_sample(sample: Dict[str, Any], max_seq_len: int) -> torch.Tensor:
    return torch.from_numpy(
        np.frombuffer(sample['tokens'],
                      dtype=np.int64)[:max_seq_len].copy())

def get_streaming_dataloader(streaming_path, tokenizer, args):
    data_collator = StreamingAutoregressiveLanguageModelDataCollator(tokenizer)
    ctx_length = args.get("seq_length", 1024) + 1  # +1 for shifting
    streaming = args.get("streaming", False)
    seed = args.get("seed", 3407)
    batch_size = args.get("batch_size", 1)

    # Provide remote support as needed
    dataset = StreamingDataset(local=streaming_path, split=None, shuffle=True)
    dataloader_streaming = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda samples: data_collator.custom_collate(samples, ctx_length)
        )
    return iter(deepspeed.utils.RepeatingLoader(dataloader_streaming))