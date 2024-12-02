from pathlib import Path

from peft import PeftModel
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
)

import torch
from torchvision.utils import draw_segmentation_masks
from torchvision.transforms.functional import to_pil_image


def get_pretrained_prompt_generator(adapter_path: str | Path) -> PeftModel:
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    processor.tokenizer.padding_side = "left"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return processor, model


def draw_colored_masks(pixel_values: torch.Tensor, masks: list[torch.Tensor]) -> torch.Tensor:
    # check if pixel values is in 0-1 range
    if pixel_values.max() > 1:
        pixel_values = pixel_values / 255.0
    elif pixel_values.min() < 0:
        pixel_values = pixel_values * 0.5 + 0.5

    
    # define
    colors = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "green": (0, 255, 0),
        "cyan": (0, 255, 255),
    }
    assert len(masks) <= len(colors), "Too many masks"
    color_names = list(colors.keys())[: len(masks)]

    # sort masks
    sorted_masks = []
    for i, mask in enumerate(masks):
        sorted_masks.append((mask, color_names[i]))
    sorted_masks.sort(key=lambda x: x[0].sum(), reverse=True)

    # draw
    for mask, color_name in sorted_masks:
        pixel_values = draw_segmentation_masks(pixel_values, mask.bool(), alpha=1, colors=colors[color_name])

    return to_pil_image(pixel_values), color_names


def output_to_color_dict(output: str, colors: list[str]):
    assistant_str = "ASSISTANT: "
    assistant_idx = output.find(assistant_str)
    if assistant_idx == -1:
        return None
    output = output[assistant_idx + len(assistant_str) :]

    color_dict = {}
    for color in colors:
        color_start = f"<{color}>"
        color_end = f"</{color}>"
        color_start_idx = output.find(color_start)
        color_end_idx = output.find(color_end)
        if color_start_idx == -1 or color_end_idx == -1:
            color_dict[color] = ""
        else:
            color_dict[color] = output[color_start_idx + len(color_start) : color_end_idx]
    
    return color_dict
