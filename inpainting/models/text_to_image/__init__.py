import torch
from .pipelines.pipeline_stable_diffusion_inpaint import (
    FreestyleStableDiffusionInpaintPipeline,
)
from pathlib import Path
from einops import repeat
from torch.nn import functional as F
from transformers import PreTrainedTokenizer
import os


def get_pretrained_mm_inpainting_pipe(
    sd2_path: str | Path, adapter_path: str | Path, device
) -> FreestyleStableDiffusionInpaintPipeline:
    use_safetensors = False
    # check if the file unet/diffusion_pytorch_model has .safetensors extension
    if os.path.exists(os.path.join(sd2_path, "unet", "diffusion_pytorch_model.safetensors")):
        use_safetensors = True
    pipe = FreestyleStableDiffusionInpaintPipeline.from_pretrained(
        sd2_path,
        torch_dtype=torch.float16,
        use_safetensors=use_safetensors,
    )
    pipe.to(device)
    pipe.load_lora_weights(adapter_path)
    return pipe


def get_freestyle_layout_mask(masks: list[torch.Tensor], max_concepts=5):
    masks = torch.stack(masks)
    background_mask = torch.all(masks == 0, dim=0)
    background_mask = repeat(background_mask, "c h w -> b c h w", b=len(masks))
    masks[background_mask] = 1
    # now background becomes all ones
    background_mask = torch.ones(background_mask.shape[1:]).unsqueeze(0).float()
    freestyle_layout_mask = torch.cat((background_mask, masks), dim=0)
    freestyle_layout_mask = freestyle_layout_mask.squeeze(1)
    # pad the layout mask to have the first dimension equal to self.max_concepts + 1
    freestyle_layout_mask = F.pad(
        freestyle_layout_mask,
        (0, 0, 0, 0, 0, max_concepts + 1 - freestyle_layout_mask.shape[0]),
        value=0,
    )
    return freestyle_layout_mask


def get_freestyle_text_inputs(texts: list[str], tokenizer: PreTrainedTokenizer):
    text_inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    text_lengths = (
        text_inputs["attention_mask"].sum(dim=1) - 2
    )  # -2 for the special tokens
    n_seps = 0
    texts_max_length = tokenizer.model_max_length - 2 - n_seps
    while (
        text_lengths.sum() > texts_max_length
        and text_lengths.max() != text_lengths.min()
    ):
        max_idx = torch.argmax(text_lengths)
        text_lengths[max_idx] -= 1
    while text_lengths.sum() > texts_max_length:
        text_lengths -= 1
    texts_mask = repeat(
        torch.arange(tokenizer.model_max_length), "l -> b l", b=len(texts)
    )
    texts_mask = (texts_mask > 0) & (texts_mask <= text_lengths.unsqueeze(1))

    # Mix the texts
    input_ids = text_inputs.input_ids
    partial_input_ids = torch.empty(0, dtype=torch.long)
    period_token_id = tokenizer.convert_tokens_to_ids(".</w>")
    for i in range(len(texts)):
        partial_input_ids = torch.cat(
            (partial_input_ids, input_ids[i][texts_mask[i]])
        )
    final_input_ids = torch.cat(
        (
            torch.tensor([tokenizer.bos_token_id]),
            partial_input_ids,
            torch.tensor([tokenizer.eos_token_id]),
        )
    )
    final_input_ids = F.pad(
        final_input_ids,
        (0, tokenizer.model_max_length - len(final_input_ids)),
        value=tokenizer.pad_token_id,
    )

    # Create the freestyle mask
    freestyle_attention_mask = torch.arange(
        1, len(texts) + 1
    ).repeat_interleave(text_lengths)
    freestyle_attention_mask = torch.cat(
        (
            torch.tensor([0]),
            freestyle_attention_mask,
            torch.tensor([0]),
        )
    )
    freestyle_attention_mask = F.pad(
        freestyle_attention_mask,
        (0, tokenizer.model_max_length - len(freestyle_attention_mask)),
        value=0,
    )
    return final_input_ids, freestyle_attention_mask
