import click
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import json
import numpy as np
import os
from .utils import is_bbox_good_image_area, create_and_blur_mask, make_collage
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from sklearn.model_selection import train_test_split


@click.group()
def cli():
    pass


def extract_masks(image: Image, entities: list[dict]) -> list[np.ndarray]:
    """Extract masks from the image based on the bounding boxes of the entities.

    Args:
        image (Image): The image.
        entities (list[dict]): The entities with their bounding boxes.

    Returns:
        list[np.ndarray]: The masks.
    """
    img_width, img_height = image.size
    img_area = img_width * img_height
    masks = []

    for k, v in entities.items():
        if not v["label"] or v["parent"] != -1:
            continue
        bounds = v["bounds"]
        bbox = [
            bounds["topLeft"]["x"],
            bounds["topLeft"]["y"],
            bounds["bottomRight"]["x"],
            bounds["bottomRight"]["y"],
        ]
        if is_bbox_good_image_area(
            img_width, img_height, bbox, 1, 65, is_normalized=False
        ):
            mask = create_and_blur_mask(
                img_width, img_height, bbox, is_normalized=False
            )
            if mask is not None:
                masks.append((k, mask))

    return masks


@cli.command()
@click.option("--annotations-dir", type=click.Path(exists=True), required=True)
@click.option("--image-dir", type=click.Path(exists=True), required=True)
@click.option("--out-dir", type=click.Path(), required=True)
def make_masks_dataset(
    annotations_dir: Path | str, image_dir: Path | str, out_dir: Path | str
):
    """Create a dataset of masks from annotations

    Args:
        annotations_dir (Path | str): _description_
        image_dir (Path | str): _description_
        out_dir (Path | str): _description_
    """
    annotations_dir = Path(annotations_dir)
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    for file_json in tqdm(list(annotations_dir.iterdir())):
        with open(file_json, "r") as f:
            data = json.load(f)
        image = Image.open(image_dir / data["image"])
        entities = data["mask_data"]
        masks = extract_masks(image, entities)
        if masks:
            image_filename = Path(image.filename)
            k_dir_path = out_dir / image_filename.stem
            os.makedirs(k_dir_path, exist_ok=True)
            for j, mask in masks:
                mask = Image.fromarray(mask)
                mask.save(k_dir_path / f"mask_{j}.png")


# Create a global lock for directory creation
directory_creation_lock = threading.Lock()


# Helper function for processing each annotation file
def _process_annotation_file(file_json, image_dir, out_dir):
    with open(file_json, "r") as f:
        data = json.load(f)
    image = Image.open(image_dir / data["image"])
    image_filename = Path(image.filename)
    image_entities_dir = out_dir / image_filename.stem
    entities = data["mask_data"]

    annotations_dict = {}
    for k, v in entities.items():
        if not v["label"] or v["parent"] != -1:
            continue
        bounds = v["bounds"]
        bbox = [
            bounds["topLeft"]["x"],
            bounds["topLeft"]["y"],
            bounds["bottomRight"]["x"],
            bounds["bottomRight"]["y"],
        ]

        concept_name = v["label"]
        concept_caption = v["caption"]

        if not is_bbox_good_image_area(
            image.width, image.height, bbox, 1, 65, is_normalized=False
        ):
            continue

        # Crop the portion of the image corresponding to the bbox
        x_min, y_min, x_max, y_max = bbox
        if x_max <= x_min or y_max <= y_min:
            continue

        cropped_image = image.crop((x_min, y_min, x_max, y_max))

        image_entities_dir.mkdir(parents=True, exist_ok=True)
        annotations_dict[f"mask_{k}"] = {
            "concept": concept_name,
            "caption": concept_caption,
        }
        cropped_image.save(image_entities_dir / f"mask_{k}.png")

    # Save annotations for this image
    if annotations_dict:
        with open(image_entities_dir / "annotations.json", "w") as f:
            json.dump(annotations_dict, f, indent=4)


@cli.command()
@click.option(
    "--annotations-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory with the annotations.",
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
    annotations_dir: Path | str, image_dir: Path | str, out_dir: Path | str
):
    """Create a dataset of entities from the annotations and images."""
    annotations_dir = Path(annotations_dir)
    image_dir = Path(image_dir)
    out_dir = Path(out_dir)

    # Get list of JSON files to process
    json_files = list(annotations_dir.iterdir())

    # Process each file in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(_process_annotation_file, file_json, image_dir, out_dir)
            for file_json in json_files
        ]

        # Use tqdm to track progress
        for future in tqdm(as_completed(futures), total=len(futures)):
            # Check if the future completed successfully
            if future.exception() is not None:
                print(f"An error occurred: {future.exception()}")


@cli.command()
@click.option("--entities-dir", type=click.Path(exists=True), required=True)
def to_lowercase(entities_dir: Path | str):
    """Convert all filenames in the entities directory to lowercase."""
    entities_dir = Path(entities_dir)

    # list all dirs in the entities directory
    dirs = [d for d in entities_dir.iterdir() if d.is_dir()]

    # open annotations.json and for each mask field, make it lowercase
    for d in tqdm(dirs):
        with open(d / "annotations.json", "r") as f:
            data = json.load(f)
        new_data = {}
        for k, v in data.items():
            for field in v:
                s = v[field]
                s = s.lower()
                s = s.replace("\n", " ")
                s = s.strip()

                # Remove special characters
                s = "".join(
                    char for char in s if char.isalnum() or char in [" ", ".", ",", "!"]
                )

                # Replace final period if present
                if s and s[-1] == ".":
                    s = s[:-1]
                v[field] = s
            new_data[k] = v
        with open(d / "annotations.json", "w") as f:
            json.dump(new_data, f, indent=4)


@cli.command()
@click.option(
    "--annotations-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory with the annotations.",
)
@click.option(
    "--image-dir",
    type=click.Path(exists=True),
    required=True,
    help="Path to the directory with the images.",
)
@click.option(
    "--split-path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the JSON split.",
)
def train_val_test_split(annotations_dir: str | Path, image_dir: str | Path, split_path: str | Path):
    """Split the images into train, validation, and test sets.

    Args:
        annotations_dir (str | Path): Path to the directory with the annotations.
        image_dir (str | Path): Path to the directory with the images.
        split_path (str | Path): Path to the JSON split.
    """
    annotations_dir = Path(annotations_dir)
    image_dir = Path(image_dir)
    split_path = Path(split_path)

    # Read splits
    with open(split_path, "r") as f:
        splits = json.load(f)

    # Make the image splits by reading the annotations
    image_splits = {}
    for split, annotations in splits.items():
        images = set()
        for annotation in annotations:
            with open(annotations_dir / annotation, "r") as f:
                data = json.load(f)
            images.add(data["image"])
        image_splits[split] = images

    # Make "train", "val", "test" directories
    for split in ["train", "val", "test"]:
        split_dir = image_dir / split
        split_dir.mkdir(exist_ok=True)

    # Rename "valid" to "val"
    if "valid" in image_splits:
        image_splits["val"] = image_splits.pop("valid")

    # Move images to the appropriate split directory
    for split, images in image_splits.items():
        for image in images:
            os.rename(image_dir / image, image_dir / split / image)
    
    # Make "unannotated" directory
    unannotated_dir = image_dir / "unannotated"
    unannotated_dir.mkdir(exist_ok=True)

    # Move images without annotations to the "unannotated" directory
    for image in image_dir.iterdir():
        if image.is_file():
            os.rename(image, unannotated_dir / image.name)


if __name__ == "__main__":
    cli()
