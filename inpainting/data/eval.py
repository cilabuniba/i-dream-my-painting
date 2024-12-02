import click
from ..models.metrics import CLIPScoreText2Image
import json
import torch
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPProcessor, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

@click.group()
def cli():
    pass


@cli.command()
@click.option("--annotations-path", type=click.Path(exists=True), required=True)
@click.option("--images-dir", type=click.Path(), required=True)
@click.option("--batch-size", type=int, default=32)
@torch.no_grad()
def evaluate_global(annotations_path, images_dir, batch_size):
    print(f"Evaluating {annotations_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    annotations_path = Path(annotations_path)
    images_dir = Path(images_dir)

    clip_score = CLIPScoreText2Image(
        text_model=CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32"),
        vision_model=CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32"),
        processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
    ).to(device)

    with open(annotations_path, "r") as f:
        annotations = json.load(f)

    batch_images = []
    batch_texts = []
    progress_bar = tqdm(annotations.items())
    for image_name, ann in annotations.items():
        # check if image is in images_dir / "train", "val", "test"
        splits = ["train", "val", "test"]
        for split in splits:
            image_path = images_dir / split / image_name
            if image_path.exists():
                break
        else:
            continue
        
        # open image
        image = Image.open(image_path)
        batch_images.append(image)

        # get text
        answer_idx = ann["description"].index("Answer: ")
        if answer_idx == -1:
            continue
        text = ann["description"][answer_idx + len("Answer: "):]
        if not isinstance(text, str):
            continue
        batch_texts.append(text)

        progress_bar.update(1)
        if len(batch_images) == batch_size:
            score = clip_score(batch_texts, batch_images)
            progress_bar.set_postfix({"score": score})
            batch_images = []
            batch_texts = []

    # process last batch
    if len(batch_images) > 0:
        score = clip_score(batch_texts, batch_images)
        progress_bar.set_postfix({"score": score})

    final_score = clip_score.compute()
    print(f"Final score: {final_score}")


@cli.command()
@click.option("--entities-dir", type=click.Path(exists=True), required=True)
@click.option("--batch-size", type=int, default=32)
@torch.no_grad()
def evaluate_local(entities_dir, batch_size):
    print(f"Evaluating {entities_dir}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    entities_dir = Path(entities_dir)

    clip_score = CLIPScoreText2Image(
        text_model=CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32"),
        vision_model=CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32"),
        processor=CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
    ).to(device)

    batch_images = []
    batch_texts = []
    progress_bar = tqdm(entities_dir.iterdir())
    for entity_dir in entities_dir.iterdir():
        # open file annotations.json if present
        annotations_path = entity_dir / "annotations.json"
        if not annotations_path.exists():
            continue
        with open(annotations_path, "r") as f:
            annotations = json.load(f)
        for k, ann in annotations.items():
            # open image
            image_path = entity_dir / f"{k}.png"
            image = Image.open(image_path)
            batch_images.append(image)

            # get text
            text = ann["caption"]
            batch_texts.append(text)

            if len(batch_images) == batch_size:
                score = clip_score(batch_texts, batch_images)
                progress_bar.set_postfix({"score": score})
                batch_images = []
                batch_texts = []
        progress_bar.update(1)
    
    # process last batch
    if len(batch_images) > 0:
        score = clip_score(batch_texts, batch_images)
        progress_bar.set_postfix({"score": score})

    final_score = clip_score.compute()
    print(f"Final score: {final_score}")


if __name__ == "__main__":
    cli()