from abc import ABC, abstractmethod
from functools import partial
from typing import Dict, Literal, Optional, Tuple, Union
import einops
import numpy as np

import torch
from torch import nn, TensorType

from jaxtyping import Int, Float
from transformer_lens import HookedTransformer, HookedTransformerConfig
from transformers import PreTrainedTokenizerBase

from src.utils import store_activations_hook, to_numpy
from transformer_lens.utils import get_offset_position_ids


# This is a mixin-class ; some other parent class is initialized by the super().__init__() call
class MooreOracle(ABC):
    # Expects vocab size as first args
    def __init__(self, *args, **kwargs):

        self.vocab_size = args[0]
        super().__init__(*args[1:], **kwargs)
        
        self.alphabet = [str(i) for i in range(self.vocab_size-1)]  # Excluding the bos_token

    @abstractmethod
    def classify_word(self, word: str | TensorType) -> int:
        return NotImplementedError("Abstract methods should not be called.")
    
    @abstractmethod
    def get_first_RState(self) -> TensorType:
        return NotImplementedError("Abstract methods should not be called.")
    
    @abstractmethod
    def get_next_RState(self, state: TensorType, char: str) -> Tuple[TensorType, int]:
        return NotImplementedError("Abstract methods should not be called.")

class RNNOracle(MooreOracle, nn.Module):
    def __init__(
            self, 
            vocab_size: int,    
            embedding_dim: int, 
            hidden_dim: int, 
            output_size: int):
        
        super().__init__(vocab_size)  

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)  # RNN expects [batch_size, vocab_size]
        self.classifier = nn.Linear(hidden_dim, output_size)     
    
    def classify_word(self, word):
        if isinstance(word, str):  
            word = [int(c) for c in word]

        x = torch.tensor(word, dtype=torch.int32)
        if len(x.shape) == 1:      # [sequence_length]
            x = x.unsqueeze(0)  # [1, sequence_length]

        if x.shape[1] > 0:
            x = self.embedding(x)       # [batch_size, sequence_length, embedding_dim]
            _, x = self.rnn(x)
        else:
            x = torch.zeros((x.shape[0], self.hidden_dim)) # h0 = zeros
        x = self.classifier(x)  
        return torch.argmax(x).item()

    def get_first_RState(self):
        first_state = torch.zeros(self.hidden_dim)
        classification = torch.argmax(self.classifier(first_state)).item()
        return to_numpy(first_state), classification
    
    def get_next_RState(self, state, char):
        state = torch.tensor(state)
        emb = self.embedding(torch.tensor([int(char)]))  # [1] -> [1, embedding_dim]
        _, new_state = self.rnn(emb, state.unsqueeze(0))         # ([1, embedding_dim], [hidden_dim]) -> (_, [hidden_dim])
        classification = torch.argmax(self.classifier(new_state)).item()
        return to_numpy(new_state.squeeze(0)), classification

# Positional Embeddings
class SinusPosEmbed(nn.Module):
    def __init__(self, cfg: Union[Dict, HookedTransformerConfig]):
        super().__init__()
        if isinstance(cfg, Dict):
            cfg = HookedTransformerConfig.from_dict(cfg)
        self.cfg = cfg

        d = cfg.d_model
        n_ctx = cfg.n_ctx
        if not hasattr(cfg, "pos_embed_n") or cfg.pos_embed_n is None:
            n = 10000
        else:
            n = cfg.pos_embed_n
            

        pos_embed_matrix = np.zeros((n_ctx, d))
        for k in range(0,n_ctx):
            for i in np.arange(int(d/2)):
                denominator = np.power(n, 2*i/d)
                pos_embed_matrix[k, 2*i] = np.sin(k/denominator)
                pos_embed_matrix[k, 2*i+1] = np.cos(k/denominator)

        self.W_pos =  nn.Parameter(
            torch.tensor(pos_embed_matrix, dtype=torch.float32),
            requires_grad=False
            )

    def forward(
        self,
        tokens: Int[torch.Tensor, "batch pos"],
        past_kv_pos_offset: int = 0,
        attention_mask: Optional[Int[torch.Tensor, "batch offset_pos"]] = None,
    ) -> Float[torch.Tensor, "batch pos d_model"]:
        """
        Forward pass for positional embeddings.

        Args:
            tokens (Int[torch.Tensor, "batch pos"]): Input tokens.
            past_kv_pos_offset (int, optional): The length of tokens in the past_kv_cache. Defaults to 0.
            attention_mask (Int[torch.Tensor, "batch pos"], optional): The attention mask for padded tokens.
                 Defaults to None.

        Returns:
            Float[torch.Tensor, "batch pos d_model"]: Absolute position embeddings.
        """
        tokens_length = tokens.size(-1)

        if attention_mask is None:
            pos_embed = self.W_pos[
                past_kv_pos_offset : tokens_length + past_kv_pos_offset, :
            ]  # [pos, d_model]
            batch_pos_embed = einops.repeat(
                pos_embed, "pos d_model -> batch pos d_model", batch=tokens.size(0)
            )

        else:
            # Separated from the no padding case for computational efficiency
            # (this code is a bit slower than the code above)

            offset_position_ids = get_offset_position_ids(
                past_kv_pos_offset, attention_mask
            )
            pos_embed = self.W_pos[offset_position_ids]  # [batch, pos, d_model]

            # Set the position embeddings to 0 for pad tokens (this is an arbitrary choice)
            padding_mask = ~attention_mask.bool()  # [batch, tokens_length]
            offset_padding_mask = padding_mask[
                :, past_kv_pos_offset : tokens_length + past_kv_pos_offset
            ].unsqueeze(
                -1
            )  # [batch, pos, 1]
            batch_pos_embed = torch.where(offset_padding_mask, 0, pos_embed)

        return batch_pos_embed.clone()

class TransformerOracle(MooreOracle, HookedTransformer):
    def __init__( 
            self,
            cfg: Dict,
            move_to_device: bool = True,
            rstate_cache = None):
        
        self.full_cfg = cfg

        # This is used by the HookedTransformer super class
        self.cfg = HookedTransformerConfig(
            n_layers = cfg["n_layers"],
            d_model = cfg["d_model"],
            n_ctx = cfg["n_ctx"],
            d_head = cfg["d_head"],
            act_fn= cfg["act_fn"],
            d_vocab=cfg["d_vocab"],
            d_vocab_out=cfg["d_vocab_out"],
            n_heads=cfg["n_heads"],
            d_mlp=cfg["d_mlp"],
            normalization_type=cfg["normalization_type"],
            positional_embedding_type=cfg["positional_embedding_type"],
        )  

        super().__init__(self.cfg.d_vocab, self.cfg, None, move_to_device)  

        if self.cfg.positional_embedding_type == "sinus":
            self.pos_embed = SinusPosEmbed(self.cfg)

        if rstate_cache is None:
            rstate_cache = f"blocks.{self.cfg.n_layers-1}.hook_resid_post"

        self.embedding_dim = self.cfg.d_model        # embedding_dim == hidden_dim (really the dimension of the state) here, but d_mlp can be different
        self.output_size = self.cfg.d_vocab_out
        self.nb_layers = self.cfg.n_layers  
    
        self.bos_token = cfg["d_vocab"] - 1 # Last token is the bos token
        self.rstate_cache = rstate_cache

        self.Rstate_dict = {}   # dict from activations (torch tensors) to the words that caused it

    def classify_word(self, word):
        if isinstance(word, str):
            word = [self.bos_token] + [int(c) for c in word]
        # if not we assume there is a bos_token present
        x = torch.tensor(word, dtype=torch.int32)

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        logits = self.forward(x, return_type="logits") 

        return torch.argmax(logits[0,-1,:], dim=-1).item()
    
    def get_first_RState(self):
        self.remove_all_hook_fns()
        acts = []

        hook = partial(store_activations_hook, store=acts)
        self.add_hook(self.rstate_cache, hook)

        classification = self.classify_word("")
    
        state = to_numpy(acts[0][0, 0, :])

        self.Rstate_dict.update({str(state): ""})

        return state, classification
    
    def get_next_RState(self, state, char):
        prev_word = self.Rstate_dict[str(state)]

        word = prev_word + char

        self.remove_all_hook_fns()
        acts = []                       # Will store a list of [batch_size, sequence_length, d_model]

        hook = partial(store_activations_hook, store=acts)
        self.add_hook(self.rstate_cache, hook)

        classification = self.classify_word(word)

        state = to_numpy(acts[0][0, -1, :])

        self.Rstate_dict.update({str(state): word})

        return state, classification
    
    def label_word(self, word):
        if isinstance(word, str):
            word = [self.bos_token] + [int(c) for c in word]
        # if not we assume there is a bos_token present
        if not isinstance(word, torch.Tensor):
            x = torch.tensor(word, dtype=torch.int32)
        else:
            x = word

        if len(x.shape) == 1:
            x = x.unsqueeze(0)

        logits = self.forward(x, return_type="logits") 

        return torch.argmax(logits[0,:,:], dim=-1).tolist()