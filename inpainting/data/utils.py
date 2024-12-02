from PIL import Image
import math
import numpy as np
import cv2


def compute_bbox_area(img_width: int, img_height: int, bbox: list[float], is_normalized: bool = True) -> float:
    """Compute the area of a bounding box.

    Args:
        img_width (int): The width of the image.
        img_height (int): The height of the image.
        bbox (list[float]): The bounding box coordinates.

    Returns:
        float: The area of the bounding box.
    """
    if is_normalized:
        x_min = bbox[0] * img_width
        y_min = bbox[1] * img_height
        x_max = bbox[2] * img_width
        y_max = bbox[3] * img_height
    else:
        x_min, y_min, x_max, y_max = bbox

    width = x_max - x_min
    height = y_max - y_min

    area = width * height
    return area


def is_bbox_good_image_area(
    img_width: int,
    img_height: int,
    bbox: list[float],
    min_threshold_percentage: float,
    max_threshold_percentage: float,
    is_normalized: bool = True,
):
    """Check if the bounding box area is within the specified thresholds.

    Args:
        img_width (int): The width of the image.
        img_height (int): The height of the image.
        bbox (list[float]): The bounding box coordinates.
        min_threshold_percentage (float): The minimum threshold percentage.
        max_threshold_percentage (float): The maximum threshold percentage.

    Returns:
        bool: True if the bounding box area is within the thresholds, False otherwise.
    """
    bbox_area = compute_bbox_area(img_width, img_height, bbox, is_normalized)
    img_area = img_width * img_height
    return (
        bbox_area <= (max_threshold_percentage / 100) * img_area
        and bbox_area >= (min_threshold_percentage / 100) * img_area
    )


def create_and_blur_mask(
    img_width: int, img_height: int, bbox: list[float], blur: bool = False, is_normalized: bool = True
) -> np.ndarray:
    """Create a mask from the bounding box and optionally blur it.

    Args:
        img_width (int): The width of the image.
        img_height (int): The height of the image.
        bbox (list[float]): The bounding box coordinates.
        blur (bool, optional): Whether to blur the mask. Defaults to False.

    Returns:
        np.ndarray: The mask.
    """
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    if is_normalized:
        x_min = int(bbox[0] * img_width)
        y_min = int(bbox[1] * img_height)
        x_max = int(bbox[2] * img_width)
        y_max = int(bbox[3] * img_height)
    else:
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]

    if x_max <= x_min or y_max <= y_min:
        return None

    mask[y_min:y_max, x_min:x_max] = 255

    if not blur:
        return mask
    return cv2.GaussianBlur(mask, (25, 25), sigmaX=0)


def make_collage(
    images: list[Image.Image], padding_color: tuple[int] = (127, 127, 127)
):
    """Create a collage from a list of images.

    Args:
        images (list[Image.Image]): The list of images to create the collage from.
        padding_color (tuple[int], optional): The color of the padding. Defaults to (127, 127, 127).

    Returns:
        Image.Image: The collage image
    """
    # Determine the number of images and best grid size
    num_images = len(images)
    grid_size = math.ceil(math.sqrt(num_images))  # Closest square grid size
    grid_cols = grid_size
    grid_rows = math.ceil(num_images / grid_cols)

    # Find maximum width and height to determine the size of each cell in the grid
    max_width = max(image.width for image in images)
    max_height = max(image.height for image in images)

    # Calculate the size of the collage
    collage_width = grid_cols * max_width
    collage_height = grid_rows * max_height

    # Create a blank canvas with padding color
    collage = Image.new("RGB", (collage_width, collage_height), padding_color)

    # Paste images into the collage
    for index, image in enumerate(images):
        row = index // grid_cols
        col = index % grid_cols
        x = col * max_width
        y = row * max_height
        collage.paste(image, (x, y))

    # Save the final collage
    return collage


def extract_masks(image: Image.Image, entities: dict) -> list[np.ndarray]:
    """Extract masks from the image based on the bounding boxes of the entities.

    Args:
        image (Image.Image): The image.
        entities (dict): The entities with their bounding boxes.

    Returns:
        list[np.ndarray]: The masks.
    """
    img_width, img_height = image.size
    masks = []

    for k, v in entities.items():
        bboxes = v["bboxes"]
        entity_masks = []
        for bbox in bboxes:
            if is_bbox_good_image_area(img_width, img_height, bbox, 1, 65):
                mask = create_and_blur_mask(img_width, img_height, bbox)
                if mask is not None:
                    entity_masks.append(mask)
        # an entity can have multiple bounding boxes
        if entity_masks:
            entity_mask = np.zeros((img_height, img_width), dtype=np.uint8)
            for mask in entity_masks:
                entity_mask = np.maximum(entity_mask, mask)
            masks.append(entity_mask)

    # Return an empty mask if no valid bounding boxes are found
    return masks


def clean_caption(caption: str) -> str:
    """Clean the caption by removing special characters and extra spaces.

    Args:
        caption (str): The caption.

    Returns:
        str: The cleaned caption.
    """
    caption = caption.lower()
    caption = caption.replace("\n", " ")

    # Remove special characters
    caption = "".join(
        char for char in caption if char.isalnum() or char in [" ", ".", ",", "!"]
    )

    # Replace final period if present
    if caption[-1] == ".":
        caption = caption[:-1]

    # Replace string "The image shows "
    caption = caption.replace("the image shows ", "")

    if not "collage" in caption:
        return caption

    special_words = [
        "featuring",
        "depicting",
        "showing",
        "containing",
        "illustrating",
        "displaying",
        "exhibiting",
        "portraying",
        "revealing",
        "highlighting",
        "showcasing",
        "demonstrating",
        "introducing",
        "exposing",
        "unveiling",
        "including",
    ]
    for word in special_words:
        if word in caption:
            word_idx = caption.index(word)
            caption = caption[word_idx + len(word + " ") :]
            break

    if "collage of" in caption:
        collage_of_idx = caption.index("collage of")
        caption = caption[collage_of_idx + len("collage of ") :]
    
    return caption


    
