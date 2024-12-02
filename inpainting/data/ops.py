import concurrent.futures
import json
import os
import sys
from glob import glob
from pathlib import Path

import click
import spacy
import torch
from PIL import Image
from safetensors.torch import load_file, save_file
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
)
import pandas as pd

from .utils import clean_caption, extract_masks, is_bbox_good_image_area, make_collage


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--annotations-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the json file with the annotations.",
)
@click.option(
    "--image-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory with the images.",
)
@click.option(
    "--out-dir", type=click.Path(), required=True, help="Path to the output directory."
)
def make_masks_dataset(
    annotations_path: Path | str, image_dir: Path | str, out_dir: Path | str
):
    """Create a dataset of masks from the annotations and images.

    The masks are extracted from the images based on the bounding boxes of the entities in the annotations.
    The masks are named as "mask_{index}.png" and saved in directories named after the images.
    There is a mask for each entity in the annotations. Multi-box entities are combined into a single mask.
    The annotations are expected to be in the following format:

    ```
    "a-mishra_expression-of-sadness-i.jpg": {
        "description": "Question: What are the details of this painting? Answer: The black blob is a cloud of black ink, which is spread across the canvas.",
        "entities": {
            "The black blob": {
                "startend": [
                    57,
                    71
                ],
                "bboxes": [
                    [
                        0.078125,
                        0.234375,
                        0.890625,
                        0.859375
                    ]
                ]
            }
        }
    }
    ```

    Args:
        annotations_path (Path | str): Path to the json file with the annotations.
        image_dir (Path | str): Path to the directory with the images.
        out_dir (Path | str): Path to the output directory.
    """
    annotations_path = Path(annotations_path)
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # if image_dir has subdirectories, make a dictionary with the name of the image and the subdirectory
    if any([f.is_dir() for f in image_dir.iterdir()]):
        image_dir_dict = {}
        for f in image_dir.iterdir():
            if f.is_dir():
                for image_file in f.iterdir():
                    image_dir_dict[image_file.name] = f.name
    else:
        image_dir_dict = None

    # open json
    with open(annotations_path, "r") as f:
        data = json.load(f)

    for k, v in tqdm(data.items()):
        if image_dir_dict is not None:
            image = Image.open(image_dir / image_dir_dict[k] / k)
        else:
            image = Image.open(image_dir / k)
        entities = v["entities"]
        masks = extract_masks(image, entities)
        if masks:
            k_dir_path = out_dir / k.replace(".jpg", "")
            os.makedirs(k_dir_path, exist_ok=True)
            for j, mask in enumerate(masks):
                mask = Image.fromarray(mask)
                mask.save(k_dir_path / f"mask_{j}.png")

# Helper function for processing each image
def process_entity(image_dir, image_dir_dict, out_dir, k, v):
    if image_dir_dict is not None:
        image_path = image_dir / image_dir_dict[k] / k
    else:
        image_path = image_dir / k

    try:
        image = Image.open(image_path)
    except Exception as e:
        return k, str(e)  # Return the key and error if image fails to open

    image_entities_dir = out_dir / k.replace(".jpg", "")
    entities = v["entities"]
    annotations_dict = {}
    j = 0

    for ek, ev in entities.items():
        bboxes = ev["bboxes"]
        entity_crops = []
        concept_name = ek.lower()
        for bbox in bboxes:
            if not is_bbox_good_image_area(image.width, image.height, bbox, 1, 65):
                continue
            # Extract the bounding box area
            x_min = int(bbox[0] * image.width)
            y_min = int(bbox[1] * image.height)
            x_max = int(bbox[2] * image.width)
            y_max = int(bbox[3] * image.height)
            if x_max <= x_min or y_max <= y_min:
                continue
            cropped_image = image.crop((x_min, y_min, x_max, y_max))
            entity_crops.append(cropped_image)

        if entity_crops:
            image_entities_dir.mkdir(parents=True, exist_ok=True)
            annotations_dict[f"mask_{j}"] = {"concept": concept_name}
            if len(entity_crops) == 1:
                entity_crops[0].save(image_entities_dir / f"mask_{j}.png")
            else:
                collage = make_collage(entity_crops)
                collage.save(image_entities_dir / f"mask_{j}.png")
            j += 1

    if j > 0:
        with open(image_entities_dir / "annotations.json", "w") as f:
            json.dump(annotations_dict, f, indent=4)

    return k, None  # Success


@cli.command()
@click.option(
    "--annotations-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the json file with the annotations.",
)
@click.option(
    "--image-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory with the images.",
)
@click.option(
    "--out-dir", type=click.Path(), required=True, help="Path to the output directory."
)
def make_entities_dataset(
    annotations_path: Path | str, image_dir: Path | str, out_dir: Path | str
):
    """Create a dataset of entities from the annotations and images."""
    annotations_path = Path(annotations_path)
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)

    with open(annotations_path, "r") as f:
        data = json.load(f)

    # Prepare a mapping for subdirectories if necessary
    image_dir_dict = None
    if any([f.is_dir() for f in image_dir.iterdir()]):
        image_dir_dict = {
            image_file.name: f.name
            for f in image_dir.iterdir()
            if f.is_dir()
            for image_file in f.iterdir()
        }

    # Use ProcessPoolExecutor to parallelize image processing
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_entity, image_dir, image_dir_dict, out_dir, k, v): k for k, v in data.items()}

        # Track progress with tqdm
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            k, error = future.result()
            if error:
                print(f"Error processing {k}: {error}")


@cli.command()
@click.option(
    "--entities-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory with the entities.",
)
@click.option(
    "--model-id",
    type=str,
    required=True,
    help="The model id for the Llava model.",
)
@click.option(
    "--batch-size",
    type=int,
    required=True,
    help="The batch size for processing the entities.",
)
@click.option(
    "--max-new-tokens",
    type=int,
    default=40,
    help="The maximum number of new tokens to generate.",
)
@click.option(
    "--num-processes",
    type=int,
    default=1,
)
@click.option(
    "--process-id",
    type=int,
    default=0,
)
def caption_masks(
    entities_dir: Path | str,
    model_id: str,
    batch_size: int,
    max_new_tokens: int = 40,
    num_processes: int = 1,
    process_id: int = 0,
):
    """Caption the masks in the entities dataset.

    The captions are generated using the Llava model from the entities.
    The captions are saved in a json file named "llava.json" in each image directory.
    The captions are saved in the following format:

    ```
    "mask_0": "The image shows a black blob."
    ```

    Args:
        entities_dir (Path | str): Path to the directory with the entities.
        model_id (str): The model id for the Llava model.
        batch_size (int): The batch size for processing the entities.
        max_new_tokens (int): The maximum number of new tokens to generate.
        num_processes (int): The number of processes to use.
        process_id (int): The process id.
    """
    entities_dir = Path(entities_dir)
    artwork_names = [f.name for f in entities_dir.iterdir() if f.is_dir()]

    # Sort the artwork names
    artwork_names = sorted(artwork_names)

    # Declare model
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    model = LlavaNextForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quantization_config,
        device_map="auto",
    )

    # count n_masks
    n_masks = 0
    for artwork_name in tqdm(artwork_names):
        artwork_entities_dir = entities_dir / artwork_name
        with open(artwork_entities_dir / "annotations.json", "r") as f:
            annotations = json.load(f)
        n_masks += len(annotations)

    print("Starting generation!")
    steps = 0
    dir_jsons = {}
    with tqdm(total=n_masks // batch_size // num_processes, file=sys.stdout) as pbar:
        batch = []
        for i, artwork_name in enumerate(artwork_names):
            if i % num_processes != process_id:
                continue
            artwork_entities_dir = entities_dir / artwork_name
            with open(artwork_entities_dir / "annotations.json", "r") as f:
                annotations = json.load(f)
            filenames = list(annotations.keys())

            for mask_name in filenames:
                raw_image = Image.open(artwork_entities_dir / f"{mask_name}.png")
                raw_image = raw_image.convert("RGB")
                concept_name = annotations[mask_name]["concept"]
                # get the artwork name from filename
                batch.append(
                    {
                        "artwork_name": artwork_name,
                        "mask_name": mask_name,
                        "raw_image": raw_image,
                        "prompt": f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWhat are the details of this image containing {concept_name} in a short sentence? Ignore the painting style. ASSISTANT: The image shows",
                        # "prompt": f"<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\n<image>\nWhat are the details of this image containing {concept_name} in a very short sentence? Ignore the painting style.<|im_end|><|im_start|>assistant\nThe image shows",
                    }
                )
                if len(batch) == batch_size:
                    steps += 1
                    images = [b["raw_image"] for b in batch]
                    prompts = [b["prompt"] for b in batch]
                    inputs = processor(
                        text=prompts, images=images, padding=True, return_tensors="pt"
                    ).to(model.device)
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        stop_strings=".",
                        tokenizer=processor.tokenizer,
                    )
                    captions = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    for j, caption in enumerate(captions):
                        an, mn = batch[j]["artwork_name"], batch[j]["mask_name"]
                        if not an in dir_jsons:
                            dir_jsons[an] = {}
                        # generated caption after word ASSISTANT:
                        keyword = "ASSISTANT: "
                        keyword_idx = caption.find(keyword)
                        dir_jsons[an][mn] = caption[keyword_idx + len(keyword) :]
                    if steps % 100 == 0:
                        dir_jsons_copy = dir_jsons.copy()
                        dir_jsons_copy["steps"] = steps
                        with open(f"backup_{process_id}.json", "w") as f:
                            json.dump(dir_jsons_copy, f, indent=4)
                    batch = []
                    pbar.update(1)

            # process the last batch
            if i + num_processes >= len(artwork_names) and batch:
                images = [b["raw_image"] for b in batch]
                prompts = [b["prompt"] for b in batch]
                inputs = processor(
                    text=prompts, images=images, padding=True, return_tensors="pt"
                ).to(model.device)
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    stop_strings=".",
                    tokenizer=processor.tokenizer,
                )
                captions = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for j, caption in enumerate(captions):
                    an, mn = batch[j]["artwork_name"], batch[j]["mask_name"]
                    if not an in dir_jsons:
                        dir_jsons[an] = {}
                    dir_jsons[an][mn] = caption[len(batch[j]["prompt"]) :].strip()
                pbar.update(1)

    # save last backup
    dir_jsons["steps"] = steps
    with open(f"backup_{process_id}_final.json", "w") as f:
        json.dump(dir_jsons, f, indent=4)


@cli.command()
@click.option(
    "--entities-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory with the entities.",
)
def extract_noun_chunk_roots(entities_dir: str | Path):
    """Extract noun chunk roots from the concepts in the entities dataset.

    The noun chunk under consideration is the first noun chunk in the concept.
    If there are no noun chunks, the concept itself is considered as the noun chunk root.
    We save these for accuracy evaluation.

    Args:
        entities_dir (str | Path): Path to the directory with the entities.
    """
    entities_dir = Path(entities_dir)
    artwork_names = [f.name for f in entities_dir.iterdir() if f.is_dir()]

    nlp = spacy.load("en_core_web_sm")

    for artwork_name in tqdm(artwork_names):
        with open(entities_dir / artwork_name / "annotations.json", "r") as f:
            annotations = json.load(f)
        for mask_name in annotations:
            concept_name = annotations[mask_name]["concept"]
            doc = nlp(concept_name)
            noun_chunks = list(doc.noun_chunks)
            if noun_chunks:
                chunk_root_text = noun_chunks[0].root.text
            else:
                chunk_root_text = concept_name
            annotations[mask_name]["noun_chunk_root"] = chunk_root_text
        with open(entities_dir / artwork_name / "annotations.json", "w") as f:
            json.dump(annotations, f, indent=4)


@cli.command()
@click.option(
    "--image-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory with the images.",
)
@click.option(
    "--masks-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory with the masks.",
)
@click.option(
    "--split-df-path",
    type=click.Path(exists=True),
    help="Path to the .CSV with the split.",
)
def train_val_test_split(image_dir: str | Path, masks_dir: str | Path, split_df_path: str | Path = None):
    """Split the images into train, validation, and test sets.

    Args:
        image_dir (str | Path): Path to the directory with the images.
        masks_dir (str | Path): Path to the directory with the masks.
        split_df_path (str | Path): Path to the .CSV with the split.
    """
    image_dir = Path(image_dir)
    masks_dir = Path(masks_dir)
    
    # Get all jpg files
    image_files = glob(str(image_dir / "*.jpg"))

    # Get all mask directories in a set
    mask_dirs = set()
    for mask_dir in masks_dir.iterdir():
        mask_dirs.add(mask_dir.name)

    # Filter out images that don't have masks
    no_annotation_image_files = [
        image_file
        for image_file in image_files
        if Path(image_file).stem not in mask_dirs
    ]
    image_files = [
        image_file for image_file in image_files if Path(image_file).stem in mask_dirs
    ]

    # Move images without annotations to a separate directory
    no_annotation_dir = image_dir / "unannotated"
    no_annotation_dir.mkdir(exist_ok=True)
    for file in no_annotation_image_files:
        os.rename(file, no_annotation_dir / Path(file).name)

    if split_df_path:
        split_df = pd.read_csv(split_df_path)
        # make the split directories
        for split in ["train", "val", "test"]:
            split_dir = image_dir / split
            split_dir.mkdir(exist_ok=True)
        # move images to the appropriate split directory
        for idx, row in split_df.iterrows():
            file_name = row["file_name"]
            split = row["split"]
            if not (image_dir / file_name).exists():
                continue
            os.rename(image_dir / file_name, image_dir / split / file_name)
    else:   
        # Split into train, val, test
        train_files, testval_files = train_test_split(
            image_files, test_size=10000, random_state=42
        )
        val_files, test_files = train_test_split(
            testval_files, test_size=5000, random_state=42
        )

        # Save the splits as different directories
        for split, files in zip(
            ["train", "val", "test"], [train_files, val_files, test_files]
        ):
            split_dir = image_dir / split
            split_dir.mkdir(exist_ok=True)
            for file in files:
                os.rename(file, split_dir / Path(file).name)


@cli.command()
@click.option(
    "--annotations-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the json file with the annotations.",
)
@click.option(
    "--out-dir", type=click.Path(), required=True, help="Path to the output directory."
)
def llava_annotations_to_folder(annotations_path: Path | str, out_dir: Path | str):
    """Create a directory structure from the annotations.

    Args:
        annotations_path (Path | str): Path to the json file with the annotations.
        out_dir (Path | str): Path to the output directory.
    """
    annotations_path = Path(annotations_path)
    out_dir = Path(out_dir)

    with open(annotations_path, "r") as f:
        data = json.load(f)

    for k in data.keys():
        if k == "steps":
            continue
        with open(out_dir / k / "llava.json", "w") as f:
            json.dump(data[k], f, indent=4)


@cli.command()
@click.option(
    "--entities-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory with the entities.",
)
def llava_annotations_to_entity_annotations(entities_dir: Path | str):
    """Create a directory structure from the annotations.

    Args:
        entities_dir (Path | str): Path to the directory with the entities.
    """
    entities_dir = Path(entities_dir)

    for k in tqdm(entities_dir.iterdir()):
        with open(k / "llava.json", "r") as f:
            llava_data = json.load(f)
        with open(k / "annotations.json", "r") as f:
            annotations = json.load(f)
        for mask_name, caption in llava_data.items():
            caption = clean_caption(caption)
            annotations[mask_name]["caption"] = caption
        with open(k / "annotations.json", "w") as f:
            json.dump(annotations, f, indent=4)


if __name__ == "__main__":
    cli()
