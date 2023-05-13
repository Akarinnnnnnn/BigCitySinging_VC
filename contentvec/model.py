import copy
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from transformers import HubertModel

# thx to @花儿不哭
class HubertModelWithFinalProj(HubertModel):
    def __init__(self, config):
        super().__init__(config)

        # The final projection layer is only used for backward compatibility.
        # Following https://github.com/auspicious3000/contentvec/issues/6
        # Remove this layer is necessary to achieve the desired outcome.
        self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

def get_hubert_model():
    model = HubertModelWithFinalProj.from_pretrained("./contentvec")
    # download from https://huggingface.co/lengyue233/content-vec-best
    model.eval()
    return model

def get_hubert_content(hmodel, wav_16k_tensor):
    feats = wav_16k_tensor
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    feats = feats.view(1, -1)
    padding_mask = torch.BoolTensor(feats.shape).fill_(False)
    #   inputs = {
    #     "source": feats.to(wav_16k_tensor.device),
    #     "padding_mask": padding_mask.to(wav_16k_tensor.device),
    #     "output_layer": 9,  # layer 9
    #   }

    logits = hmodel(feats.to(wav_16k_tensor.device))["last_hidden_state"]
    feats = hmodel.final_proj(logits)
    return feats.transpose(1,2)

