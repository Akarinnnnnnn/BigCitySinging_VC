import torch
from slicer import cut, chunks2audio
from contentvec.model import get_hubert_model, get_hubert_content

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WavProcessor:
    def __init__(self) -> None:
        self.hmodel = get_hubert_model().to(device)

    
    
