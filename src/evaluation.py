import torch
import numpy as np
from torchmetrics.functional.multimodal import clip_score
from functools import partial

class Evaluator:
    def __init__(self):
        self.clip_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")
        
    def compute_clip_score(self, image, prompt):
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0)
        # Scale to simulate a real metric score (usually around 20-30 for CLIP)
        score = self.clip_fn(image_tensor, [prompt]).item()
        return score
