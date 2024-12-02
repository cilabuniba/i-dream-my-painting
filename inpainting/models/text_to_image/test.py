import torch
from torch.utils.data import DataLoader
from diffusers import StableDiffusionInpaintPipeline
from ...data.datasets import InpaintingDataset
from .pipelines.pipeline_stable_diffusion_inpaint import (
    FreestyleStableDiffusionInpaintPipeline,
)
from PIL import Image
import click
from hydra.utils import instantiate
import yaml
import os
from tqdm import tqdm
from torchvision.transforms.functional import to_pil_image
from einops import repeat
from torchvision.transforms import GaussianBlur
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal import CLIPImageQualityAssessment
from ..metrics import CLIPScoreText2Image, CLIPScoreImage2Image
from torchvision.utils import draw_segmentation_masks
from transformers import (
    BitsAndBytesConfig,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
    CLIPProcessor,
    CLIPTokenizer,
    AutoProcessor,
    LlavaNextForConditionalGeneration,
)
from peft import PeftModel
from pathlib import Path
import json


@click.group()
def cli():
    pass


@cli.command()
@click.option("--config-path", type=click.Path(exists=True), required=True)
def generate(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = instantiate(config_dict)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.freestyle:
        pipe = FreestyleStableDiffusionInpaintPipeline.from_pretrained(
            "models/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
    pipe.to(device)

    if config.lora_path is not None:
        pipe.load_lora_weights(config.lora_path)

    dataset = config.dataset_partial(tokenizer=pipe.tokenizer)
    dataloader = config.dataloader_partial(
        dataset,
        collate_fn=dataset.collate_fn,
        worker_init_fn=lambda worker_id: dataset.generator.manual_seed(
            config.seed + worker_id
        ),
    )

    os.makedirs(config.output_dir, exist_ok=True)

    for i, batch in enumerate(tqdm(dataloader)):
        inputs = {}
        inputs["image"] = batch["pixel_values"].to(device)
        inputs["mask_image"] = batch["masks"].to(device)
        if config.freestyle:
            inputs["freestyle_attention_mask"] = batch["freestyle_attention_mask"].to(
                device
            )
            inputs["freestyle_layout_mask"] = batch["freestyle_layout_mask"].to(device)
            inputs["input_ids"] = batch["input_ids"].to(device)
        else:
            if config.fixed_prompt is not None:
                prompt = [config.fixed_prompt] * batch["pixel_values"].shape[0]
                inputs["prompt"] = prompt
            else:
                prompt_embeds = pipe.text_encoder(batch["input_ids"].to(device))
                prompt_embeds = prompt_embeds[0]
                inputs["prompt_embeds"] = prompt_embeds
        to_keep = len(batch["image_stems"])
        for image_stem in batch["image_stems"]:
            if os.path.exists(os.path.join(config.output_dir, f"{image_stem}.png")):
                to_keep -= 1
        if to_keep == 0:
            continue
        inputs = {k: v[-to_keep:] for k, v in inputs.items()}
        images = pipe(**inputs)[0]
        for image, image_stem in zip(images, batch["image_stems"]):
            image.save(os.path.join(config.output_dir, f"{image_stem}.png"))


@cli.command()
@click.option("--config-path", type=click.Path(exists=True), required=True)
def generate_prompts(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = instantiate(config_dict)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = CLIPTokenizer.from_pretrained("models/stable-diffusion-2-inpainting", subfolder="tokenizer")
    dataset: InpaintingDataset = config.dataset_partial(tokenizer=tokenizer)
    dataloader = config.dataloader_partial(
        dataset,
        collate_fn=dataset.collate_fn,
        worker_init_fn=lambda worker_id: dataset.generator.manual_seed(
            config.seed + worker_id
        ),
    )

    if not "llava_model_id" in config_dict or not "llava_output_dir" in config_dict:
        raise ValueError("llava_model_id not found in config")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
    processor.tokenizer.padding_side = "left"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-vicuna-7b-hf",
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(model, config.llava_model_id)

    os.makedirs(config.llava_output_dir, exist_ok=True)

    for batch in tqdm(dataloader):
        examples = []
        for i in range(batch["pixel_values"].shape[0]):
            example = {}
            example["pixel_values"] = batch["pixel_values"][i]
            example["all_masks"] = batch["all_masks"][i]
            example["image_stem"] = batch["image_stems"][i]
            examples.append(example)

        for example in examples:
            colors = {
                "red": (255, 0, 0),
                "green": (0, 255, 0),
                "blue": (0, 0, 255),
                "yellow": (255, 255, 0),
                "cyan": (0, 255, 255),
            }
            color_keys = list(colors.keys())
            mask_colors = [color_keys[i] for i in range(len(example["all_masks"]))][
                ::-1
            ]
            sorted_masks_with_colors = sorted(
                zip(example["all_masks"], mask_colors),
                key=lambda x: x[0].sum(),
                reverse=True,
            )
            masked_image = (example["pixel_values"] * 0.5) + 0.5
            for i, (mask, color) in enumerate(sorted_masks_with_colors):
                masked_image = draw_segmentation_masks(
                    masked_image,
                    mask.to(dtype=torch.bool),
                    alpha=1.0,
                    colors=colors[color],
                )

            # call the model
            masked_image = to_pil_image(masked_image)
            prompt = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDiscover and describe the objects hidden behind the masks of the colors: {', '.join(mask_colors)}. ASSISTANT:"
            inputs = processor(
                text=prompt,
                images=masked_image,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                stop_strings="</red>",
                tokenizer=processor.tokenizer,
            )
            output = processor.decode(output[0], skip_special_tokens=True)

            color2output = {}
            for color in mask_colors:
                color_start = output.find(f"<{color}>")
                color_end = output.find(f"</{color}>")
                if color_start == -1 or color_end == -1:
                    color_output = ""
                else: 
                    color_output = output[color_start + len(f"<{color}>") : color_end]
                color2output[color] = color_output

            texts = list(color2output.values())
            with open(
                os.path.join(
                    config.llava_output_dir, f"{example['image_stem']}.json"
                ),
                "w",
            ) as f:
                json.dump(texts, f)


@cli.command()
@click.option("--config-path", type=click.Path(exists=True), required=True)
@click.option("--only-multi-prompt", is_flag=True)
def compute(config_path, only_multi_prompt):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = instantiate(config_dict)

    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if config.freestyle:
        pipe = FreestyleStableDiffusionInpaintPipeline.from_pretrained(
            "models/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
    else:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-inpainting",
            torch_dtype=torch.float16,
        )
    pipe.to(device)

    dataset: InpaintingDataset = config.dataset_partial(tokenizer=pipe.tokenizer)
    dataloader = config.dataloader_partial(
        dataset,
        collate_fn=dataset.collate_fn,
        worker_init_fn=lambda worker_id: dataset.generator.manual_seed(
            config.seed + worker_id
        ),
    )
    del pipe

    clip_text_model = CLIPTextModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)
    clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
        "openai/clip-vit-base-patch32"
    ).to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    psnr = PeakSignalNoiseRatio(data_range=(0, 1)).to(device)
    lpips = LearnedPerceptualImagePatchSimilarity(net_type="squeeze").to(device)
    clip_score_text2image = CLIPScoreText2Image(
        text_model=clip_text_model,
        vision_model=clip_vision_model,
        processor=clip_processor,
    ).to(device)
    n_masks_psnr_scores = [
        PeakSignalNoiseRatio(data_range=(0, 1)).to(device) for _ in range(5)
    ]
    n_masks_clip_scores = [
        CLIPScoreText2Image(
            text_model=clip_text_model,
            vision_model=clip_vision_model,
            processor=clip_processor,
        ).to(device)
        for _ in range(5)
    ]
    clip_score_image2image = CLIPScoreImage2Image(
        vision_model=clip_vision_model, processor=clip_processor
    ).to(device)
    clip_iqa = CLIPImageQualityAssessment().to(device)

    gaussian_blur = GaussianBlur(kernel_size=15, sigma=20)

    for i, batch in enumerate(tqdm(dataloader)):
        pixel_values = batch["pixel_values"].to(device)

        pred_pixel_values = []
        for image_stem in batch["image_stems"]:
            pred_pixel_values.append(
                Image.open(
                    os.path.join(config.output_dir, f"{image_stem}.png")
                ).convert("RGB")
            )
        pred_pixel_values = torch.stack(
            [dataset.transforms(p) for p in pred_pixel_values]
        ).to(device)

        all_masks = batch["all_masks"]
        texts = batch["gt_texts"]

        if only_multi_prompt:
            new_pixel_values = []
            new_pred_pixel_values = []
            new_texts = []
            new_all_masks = []
            for j, example_texts in enumerate(batch["gt_texts"]):
                if len(example_texts) > 1:
                    new_pixel_values.append(pixel_values[j])
                    new_pred_pixel_values.append(pred_pixel_values[j])
                    new_texts.append(example_texts)
                    new_all_masks.append(all_masks[j])
            if len(new_pixel_values) == 0:
                continue
            pixel_values = torch.stack(new_pixel_values)
            pred_pixel_values = torch.stack(new_pred_pixel_values)
            texts = new_texts
            all_masks = new_all_masks

        # from [-1, 1] to [0, 1]
        pixel_values_01 = (pixel_values * 0.5) + 0.5
        pred_pixel_values_01 = (pred_pixel_values * 0.5) + 0.5

        # compute metrics
        psnr.update(pred_pixel_values_01, pixel_values_01)
        for j, example_masks in enumerate(all_masks):
            n_masks_psnr_scores[len(example_masks) - 1].update(pred_pixel_values_01[j], pixel_values_01[j])
        lpips.update(pred_pixel_values, pixel_values)
        fid.update(pixel_values_01, real=True)
        fid.update(pred_pixel_values_01, real=False)
        clip_iqa.update(pred_pixel_values_01)

        # compute per-image/per-mask clip scores
        for j, example_masks in enumerate(all_masks):
            example_clip_preds = []
            example_clip_refs = []
            for mask in example_masks:
                # mask is in [0, 1] of shape [1, 512, 512], we reduce the brightness of the background using the mask
                mask = mask.to(device)

                pixel_values_to_edit = pixel_values_01[j]
                pred_pixel_values_to_edit = pred_pixel_values_01[j]

                # reduce intensity of pixel_values_to_edit to 0.1
                pixel_values_to_edit = pixel_values_to_edit * 0.1
                pred_pixel_values_to_edit = pred_pixel_values_to_edit * 0.1

                # gaussian blur the pixel_values_to_edit
                pixel_values_to_edit = gaussian_blur(pixel_values_to_edit)
                pred_pixel_values_to_edit = gaussian_blur(pred_pixel_values_to_edit)

                # put the fg_pixel_values back
                fg_mask = repeat((mask > 0).squeeze(0), "h w -> c h w", c=3)
                pixel_values_to_edit[fg_mask] = pixel_values_01[j][fg_mask]
                pred_pixel_values_to_edit[fg_mask] = pred_pixel_values_01[j][fg_mask]

                example_clip_preds.append(pred_pixel_values_to_edit)
                example_clip_refs.append(pixel_values_to_edit)
            example_texts = [f"A padded image of {text}" for text in texts[j]]
            example_clip_preds = [to_pil_image(p) for p in example_clip_preds]
            example_clip_refs = [to_pil_image(p) for p in example_clip_refs]
            clip_score_text2image.update(example_texts, example_clip_preds)
            clip_score_image2image.update(example_clip_refs, example_clip_preds)
            n_masks_clip_scores[len(example_masks) - 1].update(example_texts, example_clip_preds)

    res = {
        "psnr": psnr.compute().item(),
        "lpips": lpips.compute().item(),
        "fid": fid.compute().item(),
        "clip_iqa": clip_iqa.compute().mean().item(),
        "clip_score_text2image": clip_score_text2image.compute().item(),
        "clip_score_image2image": clip_score_image2image.compute().item(),
    }

    for i in range(5):
        res[f"n_masks_psnr_{i + 1}"] = n_masks_psnr_scores[i].compute().item()
        res[f"n_masks_clip_score_text2image_{i + 1}"] = n_masks_clip_scores[i].compute().item()

    with open(
        os.path.join(
            config.output_dir, f"metrics{'_multi' if only_multi_prompt else ''}.json"
        ),
        "w",
    ) as f:
        json.dump(res, f)


if __name__ == "__main__":
    cli()
