from typing import List, Union

import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image
from torchmetrics import Metric
from transformers import (
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)


class CLIPScoreText2Image(Metric):
    def __init__(
        self,
        text_model: CLIPTextModelWithProjection,
        vision_model: CLIPVisionModelWithProjection,
        processor: CLIPProcessor,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.text_model = text_model
        self.vision_model = vision_model
        self.processor = processor

        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(
        self,
        texts: List[str],
        images: list[Image.Image],
    ) -> None:
        inputs = self.processor(
            text=texts, images=images, return_tensors="pt", size=(224, 224), padding=True, truncation=True
        ).to(self.text_model.device)

        text_embeds = self.text_model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        ).text_embeds
        image_embeds = self.vision_model(
            pixel_values=inputs["pixel_values"]
        ).image_embeds
        
        self.score += F.cosine_similarity(image_embeds, text_embeds).sum()
        self.n_samples += image_embeds.shape[0]

    def compute(self) -> torch.Tensor:
        return self.score / self.n_samples


class CLIPScoreImage2Image(Metric):
    def __init__(
        self,
        vision_model: CLIPVisionModelWithProjection,
        processor: CLIPProcessor,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_model = vision_model
        self.processor = processor

        self.add_state("score", torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(
            "n_samples", torch.tensor(0, dtype=torch.long), dist_reduce_fx="sum"
        )

    def update(
        self,
        images1: list[Image.Image],
        images2: list[Image.Image],
    ) -> None:
        inputs1 = self.processor(
            images=images1, return_tensors="pt", size=(224, 224)
        ).to(self.vision_model.device)
        inputs2 = self.processor(
            images=images2, return_tensors="pt", size=(224, 224)
        ).to(self.vision_model.device)

        image_embeds1 = self.vision_model(**inputs1).image_embeds
        image_embeds2 = self.vision_model(**inputs2).image_embeds

        self.score += F.cosine_similarity(image_embeds1, image_embeds2).sum()
        self.n_samples += image_embeds1.shape[0]

    def compute(self) -> torch.Tensor:
        return self.score / self.n_samples
