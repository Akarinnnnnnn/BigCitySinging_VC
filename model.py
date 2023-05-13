import copy
import random
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

class MelModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()


    def forward(self, hubert_vector, speaker_id):
        raise NotImplementedError