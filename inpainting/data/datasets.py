import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageDraw
import os
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from torchvision.utils import draw_segmentation_masks
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from ..models.image_to_text.schedulers import AlphaScheduler
from einops import repeat


class InpaintingDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        data_dir: str | Path,
        tokenizer: PreTrainedTokenizer,
        max_concepts: int = 5,
        generator: Optional[torch.Generator] = None,
        shuffle_concepts: bool = False,
        masked_area_threshold: Optional[float] = None,
        resolution: int = 512,
        freestyle: bool = False,
        drop_caption_probability: float = 0.0,
        override_fn=None,
        texts_dir=None,
        split: str = "train",
    ):
        """Dataset for text-to-image inpainting.

        Args:
            data_dir (str | Path): Root directory of the dataset. It should contain the following subdirectories:
                - images: contains the images in the split (train, val, test)
                - masks: contains the masks for the images
                - entities: contains the entities for the images
            tokenizer (PreTrainedTokenizer): Tokenizer to tokenize the prompts.
            max_concepts (int, optional): Maximum number of masks for multi-mask inpainting. Defaults to 5.
            generator (Optional[torch.Generator], optional): A torch generator for shuffling. Defaults to None.
            shuffle_concepts (bool, optional): Whether if masks should be shuffled. Defaults to False.
            masked_area_threshold (Optional[float], optional): A threshold of masked area above which masks will not be included.
                If the first mask is the product of multiple bounding boxes, it might be included even if the area is larger, to
                include at least one annotation. Defaults to None.
            resolution (int, optional): Image resolution. Defaults to 512.
            freestyle (bool, optional): Whether to return the `freestyle_attention_mask` and `freestyle_layout_mask`. Defaults to False.
            drop_caption_probability (float, optional): Drop single-mask examples conditioning for CFG. Defaults to 0.0.
            override_fn (_type_, optional): Override the example (unused). Defaults to None.
            texts_dir (_type_, optional): Specify a directory from which reading the prompts (for generated prompts). Defaults to None.
            split (str, optional): Specify the split. Defaults to "train".
        """
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_concepts = max_concepts
        self.generator = generator
        self.shuffle_concepts = shuffle_concepts
        self.masked_area_threshold = masked_area_threshold
        self.freestyle = freestyle
        self.drop_caption_probability = drop_caption_probability
        self.override_fn = override_fn
        self.texts_dir = texts_dir
        self.split = split

        assert self.data_dir.exists(), f"Data directory not found: {self.data_dir}"
        assert self.max_concepts is None or (
            self.max_concepts > 0 and self.max_concepts <= 10
        ), f"Invalid max_concepts: {self.max_concepts}"
        assert self.split in ["train", "val", "test"], f"Invalid split: {self.split}"

        if self.texts_dir is not None:
            if not os.path.exists(self.texts_dir):
                self.texts_dir = None
                print(f"Texts directory not found: {self.texts_dir}")

        self.images_dir = self.data_dir / "images" / self.split
        self.masks_dir = self.data_dir / "masks"
        self.entities_dir = self.data_dir / "entities"

        self.image_paths = list(self.images_dir.glob("*.jpg"))
        self.image_stems = [img.stem for img in self.image_paths]

        # remove image paths if there isn't any annotation
        self.image_paths = [
            img_path
            for img_path in self.image_paths
            if (self.entities_dir / img_path.stem / "annotations.json").exists()
        ]
        self.image_stems = [img.stem for img in self.image_paths]

        # Define transforms
        self.transforms = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.CenterCrop(resolution),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(
                    resolution, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(resolution),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def get_image_concepts2masks(self, image_name: str):
        image_features_dir = self.features_dir / image_name
        with open(image_features_dir / "annotations.json", "r") as f:
            annotations = json.load(f)
        concepts2masks = {}
        for mask_name, mask_annotations in annotations.items():
            concept = mask_annotations["concept"]
            if not concept in concepts2masks:
                concepts2masks[concept] = []
            concepts2masks[concept].append(mask_name)
        return concepts2masks

    def __getitem__(self, idx: int):
        example = {}

        image_path = self.image_paths[idx]
        image_stem = self.image_stems[idx]

        # read the image
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        example["pixel_values"] = self.transforms(image)

        # read the annotations
        with open(self.entities_dir / image_stem / "annotations.json", "r") as f:
            image_anns = json.load(f)

        # read the masks
        for mask_stem in image_anns:
            mask = Image.open(self.masks_dir / image_stem / f"{mask_stem}.png")
            if not mask.mode == "L":
                mask = mask.convert("L")
            image_anns[mask_stem]["mask"] = pil_to_tensor(mask) / 255.0

        # sort by area
        image_anns = dict(
            sorted(
                image_anns.items(),
                key=lambda x: x[1]["mask"].sum().item(),
                reverse=True,
            )
        )

        # keep only max_concepts
        image_anns = list(image_anns.items())
        if self.max_concepts is not None:
            image_anns = image_anns[: self.max_concepts]
        image_anns = dict(image_anns)

        # shuffle concepts
        if self.shuffle_concepts:
            # shuffle integers between 0 and len(image_anns) - 1
            shuffled_indices = torch.randperm(len(image_anns), generator=self.generator)
            image_anns = dict(
                zip(
                    [list(image_anns.keys())[i] for i in shuffled_indices],
                    [list(image_anns.values())[i] for i in shuffled_indices],
                )
            )

        # transform the masks
        for ann in image_anns.values():
            ann["mask"] = self.mask_transforms(ann["mask"])

        # keep only masks if their sum is under a threshold
        if self.masked_area_threshold is not None:
            to_keep = 1
            total_mask = torch.zeros_like(list(image_anns.values())[0]["mask"])
            total_pixels = total_mask.numel()
            for i, ann in enumerate(image_anns.values()):
                total_mask = torch.max(total_mask, ann["mask"])
                if i == 0:
                    continue
                if total_mask.sum().item() / total_pixels <= self.masked_area_threshold:
                    to_keep += 1
                else:
                    break
            image_anns = dict(list(image_anns.items())[:to_keep])

        # Make the mask
        masks_list = [ann["mask"] for ann in image_anns.values()]
        masks = torch.stack(masks_list)
        mask = torch.any(masks, dim=0).float()
        example["masks"] = mask

        # Make the freestyle layout if needed
        if self.freestyle:
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
                (0, 0, 0, 0, 0, self.max_concepts + 1 - freestyle_layout_mask.shape[0]),
                value=0,
            )
            example["freestyle_layout_mask"] = freestyle_layout_mask

        gt_texts = [image_anns[ann]["caption"] for ann in image_anns]

        # Read texts if needed
        if self.texts_dir is not None:
            with open(os.path.join(self.texts_dir, f"{image_stem}.json"), "r") as f:
                captions = json.load(f)
            for image_ann, caption in zip(image_anns.values(), captions):
                image_ann["caption"] = caption

        # Get the prompt
        texts = [image_anns[ann]["caption"] for ann in image_anns]
        if len(texts) == 1 and self.drop_caption_probability > 0:
            # If there is only one text, we drop it with a certain probability (use torch), and replace with ""
            if torch.rand(1).item() < self.drop_caption_probability:
                texts = [""]
        text_inputs = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_lengths = (
            text_inputs["attention_mask"].sum(dim=1) - 2
        )  # -2 for the special tokens
        n_seps = 0 if self.freestyle else len(texts)
        texts_max_length = self.tokenizer.model_max_length - 2 - n_seps
        while (
            text_lengths.sum() > texts_max_length
            and text_lengths.max() != text_lengths.min()
        ):
            max_idx = torch.argmax(text_lengths)
            text_lengths[max_idx] -= 1
        while text_lengths.sum() > texts_max_length:
            text_lengths -= 1
        texts_mask = repeat(
            torch.arange(self.tokenizer.model_max_length), "l -> b l", b=len(texts)
        )
        texts_mask = (texts_mask > 0) & (texts_mask <= text_lengths.unsqueeze(1))

        # Mix the texts
        input_ids = text_inputs.input_ids
        partial_input_ids = torch.empty(0, dtype=torch.long)
        period_token_id = self.tokenizer.convert_tokens_to_ids(".</w>")
        for i in range(len(texts)):
            partial_input_ids = torch.cat(
                (partial_input_ids, input_ids[i][texts_mask[i]])
            )
            if not self.freestyle:
                partial_input_ids = torch.cat(
                    (partial_input_ids, torch.tensor([period_token_id]))
                )
        final_input_ids = torch.cat(
            (
                torch.tensor([self.tokenizer.bos_token_id]),
                partial_input_ids,
                torch.tensor([self.tokenizer.eos_token_id]),
            )
        )
        final_input_ids = F.pad(
            final_input_ids,
            (0, self.tokenizer.model_max_length - len(final_input_ids)),
            value=self.tokenizer.pad_token_id,
        )
        example["input_ids"] = final_input_ids
        example["gt_texts"] = gt_texts
        example["texts"] = texts

        # Create the freestyle mask
        if self.freestyle:
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
                (0, self.tokenizer.model_max_length - len(freestyle_attention_mask)),
                value=0,
            )
            example["freestyle_attention_mask"] = freestyle_attention_mask

        # Get the masked image
        broadcasted_mask = mask.expand_as(example["pixel_values"])
        example["masked_pixel_values"] = example["pixel_values"] * (
            1 - broadcasted_mask
        )

        # Additional information for test
        if self.split in ["val", "test"]:
            example["image_stems"] = image_stem
            example["all_masks"] = masks_list

        if self.override_fn is not None:
            example = self.override_fn(example)

        return example

    @staticmethod
    def collate_fn(batch):
        collated = {}

        pixel_values = (
            torch.stack([example["pixel_values"] for example in batch])
            .to(memory_format=torch.contiguous_format)
            .float()
        )
        collated["pixel_values"] = pixel_values
        masks = (
            torch.stack([example["masks"] for example in batch])
            .to(memory_format=torch.contiguous_format)
            .float()
        )
        collated["masks"] = masks
        masked_pixel_values = (
            torch.stack([example["masked_pixel_values"] for example in batch])
            .to(memory_format=torch.contiguous_format)
            .float()
        )
        collated["masked_pixel_values"] = masked_pixel_values
        input_ids = torch.stack([example["input_ids"] for example in batch])
        collated["input_ids"] = input_ids
        gt_texts = [example["gt_texts"] for example in batch]
        collated["gt_texts"] = gt_texts
        texts = [example["texts"] for example in batch]
        collated["texts"] = texts
        if "freestyle_layout_mask" in batch[0]:
            freestyle_layout_mask = (
                torch.stack([example["freestyle_layout_mask"] for example in batch])
                .to(memory_format=torch.contiguous_format)
                .long()
            )
            collated["freestyle_layout_mask"] = freestyle_layout_mask
            freestyle_attention_mask = torch.stack(
                [example["freestyle_attention_mask"] for example in batch]
            )
            collated["freestyle_attention_mask"] = freestyle_attention_mask
        if "image_stems" in batch[0]:
            image_stems = [example["image_stems"] for example in batch]
            collated["image_stems"] = image_stems
            all_masks = [example["all_masks"] for example in batch]
            collated["all_masks"] = all_masks

        # if there are other keys in batch[0], add them as lists
        for key in batch[0].keys():
            if key not in collated:
                collated[key] = [example[key] for example in batch]
        return collated


class LlavaDataset(Dataset):
    """Dataset for prompt generation.

    """
    def __init__(
        self,
        data_dir: str | Path,
        max_concepts: Optional[int] = 5,
        generator: Optional[torch.Generator] = None,
        remove_intersections: bool = False,
        shuffle_concepts: bool = False,
        masked_area_threshold: Optional[float] = None,
        return_entity_PILs: bool = False,
        only_gray_concept: bool = False,
        override_gray: bool = False,
        split: str = "train",
    ):
        """Dataset for prompt generation.

        Args:
            data_dir (str | Path): Data directory. It should contain the following subdirectories:
                - images: contains the images in the split (train, val, test)
                - masks: contains the masks for the images
                - entities: contains the entities for the images
            max_concepts (Optional[int], optional): Maximum number of masks for multi-mask inpainting. Defaults to 5.
            generator (Optional[torch.Generator], optional): A torch generator for shuffling. Defaults to None.
            remove_intersections (bool, optional): Removes intersection between masks. Unused. Defaults to False.
            shuffle_concepts (bool, optional): Whether to shuffle the masks. Defaults to False.
            masked_area_threshold (Optional[float], optional): An area threshold over which masks will not be included. Defaults to None.
            return_entity_PILs (bool, optional): Whether to return the PIL images of the crops under the mask. Defaults to False.
            only_gray_concept (bool, optional): Whether to return a single concept with color gray. Defaults to False.
            override_gray (bool, optional): Whether to make the single gray concept, red. Defaults to False.
            split (str, optional): Data split. Defaults to "train".
        """
        self.data_dir = Path(data_dir)
        self.max_concepts = max_concepts
        self.generator = generator
        self.remove_intersections = remove_intersections
        self.shuffle_concepts = shuffle_concepts
        self.masked_area_threshold = masked_area_threshold
        self.return_PIL_entities = return_entity_PILs
        self.only_gray_concept = only_gray_concept
        self.override_gray = override_gray
        self.split = split

        # we add support for 10 concepts, but we use 5 with the first 5 colors
        self.colors = {
            "red": (255, 0, 0),
            "green": (0, 255, 0),
            "blue": (0, 0, 255),
            "yellow": (255, 255, 0),
            "cyan": (0, 255, 255),
            "magenta": (255, 0, 255),
            "orange": (255, 165, 0),
            "purple": (128, 0, 128),
            "pink": (255, 192, 203),
            "brown": (165, 42, 42),
        }

        assert self.data_dir.exists(), f"Data directory not found: {self.data_dir}"
        assert self.max_concepts is None or (
            self.max_concepts > 0 and self.max_concepts <= len(self.colors)
        ), f"Invalid max_concepts: {self.max_concepts}"
        assert self.split in ["train", "val", "test"], f"Invalid split: {self.split}"

        self.images_dir = self.data_dir / "images" / self.split
        self.masks_dir = self.data_dir / "masks"
        self.entities_dir = self.data_dir / "entities"

        self.image_paths = list(self.images_dir.glob("*.jpg"))
        self.image_stems = [img.stem for img in self.image_paths]

        # remove image paths if there isn't any annotation
        self.image_paths = [
            img_path
            for img_path in self.image_paths
            if (self.entities_dir / img_path.stem / "annotations.json").exists()
        ]
        self.image_stems = [img.stem for img in self.image_paths]

        # keep only max_concepts colors
        self.colors = dict(list(self.colors.items())[: self.max_concepts])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx_alphas):
        try:
            idx, alphas = idx_alphas
        except TypeError:
            idx = idx_alphas
            alphas = torch.ones(self.max_concepts)

        image_path = self.image_paths[idx]
        image_stem = self.image_stems[idx]

        # read the image
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # read the annotations
        with open(self.entities_dir / image_stem / "annotations.json", "r") as f:
            image_anns = json.load(f)

        # read the masks
        for mask_stem in image_anns:
            mask = Image.open(self.masks_dir / image_stem / f"{mask_stem}.png")
            if not mask.mode == "L":
                mask = mask.convert("L")
            image_anns[mask_stem]["mask"] = pil_to_tensor(mask) / 255.0

        # sort by area
        image_anns = dict(
            sorted(
                image_anns.items(),
                key=lambda x: x[1]["mask"].sum().item(),
                reverse=True,
            )
        )

        # keep only max_concepts
        image_anns = list(image_anns.items())
        if self.max_concepts is not None:
            image_anns = image_anns[: self.max_concepts]
        image_anns = dict(image_anns)

        # sample len(image_anns) colors
        sampled_color_keys = []
        color_keys = list(self.colors.keys())
        rand_indices = torch.randperm(len(color_keys), generator=self.generator)[
            : len(image_anns)
        ]
        for i, (mask_stem, ann) in enumerate(image_anns.items()):
            sampled_color_keys.append(color_keys[rand_indices[i]])
            ann["color_key"] = color_keys[rand_indices[i]]
            ann["color"] = self.colors[ann["color_key"]]

        # shuffle concepts
        if self.shuffle_concepts:
            # shuffle integers between 0 and len(image_anns) - 1
            shuffled_indices = torch.randperm(len(image_anns), generator=self.generator)
            image_anns = dict(
                zip(
                    [list(image_anns.keys())[i] for i in shuffled_indices],
                    [list(image_anns.values())[i] for i in shuffled_indices],
                )
            )

        # keep only masks if their sum is under a threshold
        if self.masked_area_threshold is not None:
            to_keep = 1
            total_mask = torch.zeros_like(list(image_anns.values())[0]["mask"])
            total_pixels = total_mask.numel()
            for i, ann in enumerate(image_anns.values()):
                total_mask = torch.max(total_mask, ann["mask"])
                if i == 0:
                    continue
                if total_mask.sum().item() / total_pixels <= self.masked_area_threshold:
                    to_keep += 1
                else:
                    break
            image_anns = dict(list(image_anns.items())[:to_keep])
            sampled_color_keys = sampled_color_keys[:to_keep]

            # re-sort the image_anns by area
            image_anns = dict(
                sorted(
                    image_anns.items(),
                    key=lambda x: x[1]["mask"].sum().item(),
                    reverse=True,
                )
            )

        # if split is "val", one of the annotations is always red
        if self.split in ["val", "test"] and not "red" in sampled_color_keys:
            red_idx = torch.randint(0, len(image_anns), (1,), generator=self.generator)
            image_anns[list(image_anns.keys())[red_idx]]["color_key"] = "red"
            image_anns[list(image_anns.keys())[red_idx]]["color"] = self.colors["red"]

        # if split is "val", and only_gray_concept is True, put gray instead of red
        if self.only_gray_concept:
            if self.split in ["val", "test"]:
                for ann in image_anns.values():
                    if ann["color_key"] == "red":
                        ann["color_key"] = "gray"
                        if self.override_gray:
                            ann["color"] = (255, 0, 0)
                        else:
                            ann["color"] = (128, 128, 128)
            else:
                # make the first ann gray
                list(image_anns.values())[0]["color_key"] = "gray"
                list(image_anns.values())[0]["color"] = (128, 128, 128)

        # remove intersections
        if self.remove_intersections:
            image_anns_keys = list(image_anns.keys())
            for i, key_i in enumerate(image_anns_keys):
                for key_j in image_anns_keys[i + 1 :]:
                    image_anns[key_i]["mask"] = image_anns[key_i]["mask"] * (
                        1 - image_anns[key_j]["mask"]
                    )

        # draw the masks
        masked_image = pil_to_tensor(image).clone()
        for i, (mask_stem, ann) in enumerate(image_anns.items()):
            if self.only_gray_concept and ann["color_key"] != "gray":
                continue
            color = ann["color"]
            mask = ann["mask"]
            masked_image = draw_segmentation_masks(
                masked_image,
                mask.to(dtype=torch.bool),
                alpha=alphas[i].item(),
                colors=color,
            )
        masked_image = to_pil_image(masked_image)

        # shuffle concepts
        if self.shuffle_concepts:
            # shuffle integers between 0 and len(image_anns) - 1
            shuffled_indices = torch.randperm(len(image_anns), generator=self.generator)
            image_anns = dict(
                zip(
                    [list(image_anns.keys())[i] for i in shuffled_indices],
                    [list(image_anns.values())[i] for i in shuffled_indices],
                )
            )

        # if split is "val", put the red annotation to the end
        if self.split in ["val", "test"] or self.only_gray_concept:
            sort_key = "gray" if self.only_gray_concept else "red"
            image_anns = dict(
                sorted(
                    image_anns.items(),
                    key=lambda x: x[1]["color_key"] == sort_key,
                )
            )

        if self.only_gray_concept:
            # keep only the last image_ann
            image_anns = dict(list(image_anns.items())[-1:])

        # make the target sequence
        target_sequence = ""
        for i, (mask_stem, ann) in enumerate(image_anns.items()):
            target_sequence += f"<{ann['color_key']}>"
            target_sequence += ann["caption"]
            target_sequence += f"</{ann['color_key']}>"

        example = {
            "PIL_images": image,
            "PIL_masked_images": masked_image,
            "entity_captions": [ann["caption"] for ann in image_anns.values()],
            "entity_concepts": [ann["concept"] for ann in image_anns.values()],
            "entity_colors": [ann["color_key"] for ann in image_anns.values()],
            "entity_noun_chunk_roots": [
                ann["noun_chunk_root"] for ann in image_anns.values()
            ],
            "targets": target_sequence,
        }

        # get the entity images
        if self.return_PIL_entities:
            entity_images = []
            # read the entities
            for mask_stem in image_anns:
                entity = Image.open(self.entities_dir / image_stem / f"{mask_stem}.png")
                if not entity.mode == "RGB":
                    entity = entity.convert("RGB")
                entity_images.append(entity)
            example["entity_PILs"] = entity_images

        return example

    @staticmethod
    def collate_fn(
        examples,
        processor,
        max_length: int,
        overwrite_prompt: Optional[str] = None,
        split: str = "train",
    ):
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        batch = {}

        # prepare the multimodal prompts
        masked_images = []
        prompt_prefixes = []
        prompts = []
        for example in examples:
            masked_images.append(example["PIL_masked_images"])
            # TODO: in the future we can replace this by processor.apply_chat_template
            prompt_prefix = f"A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nDiscover and describe the objects hidden behind the masks of the colors: {', '.join(example['entity_colors'])}. ASSISTANT:"
            prompt = f"{prompt_prefix} {example['targets']}"
            prompt_prefixes.append(prompt_prefix)
            prompts.append(prompt)

        if overwrite_prompt is not None:
            prompt_prefixes = [overwrite_prompt] * len(prompts)

        # process the multimodal prompts
        # check the truncation
        prefix_inputs = processor(
            text=prompt_prefixes,
            images=None if split == "train" else masked_images,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        # prepare the labels for training
        if split == "train":
            model_inputs = processor(
                text=prompts,
                images=masked_images,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            labels = model_inputs["input_ids"].clone()
            labels[labels == processor.tokenizer.pad_token_id] = (
                -100
            )  # ignore the padding tokens
            # ignore the prefix
            prefix_mask = prefix_inputs["attention_mask"]
            prefix_mask = F.pad(
                prefix_mask,
                (0, model_inputs["attention_mask"].shape[1] - prefix_mask.shape[1]),
                value=0,
            )
            labels[prefix_mask == 1] = -100
            model_inputs["labels"] = labels

        batch["model_inputs"] = model_inputs if split == "train" else prefix_inputs

        # other things needed
        batch["PIL_images"] = [example["PIL_images"] for example in examples]
        batch["PIL_masked_images"] = [
            example["PIL_masked_images"] for example in examples
        ]
        batch["entity_captions"] = [example["entity_captions"] for example in examples]
        batch["entity_concepts"] = [example["entity_concepts"] for example in examples]
        batch["entity_colors"] = [example["entity_colors"] for example in examples]
        batch["entity_noun_chunk_roots"] = [
            example["entity_noun_chunk_roots"] for example in examples
        ]
        if "entity_PILs" in examples[0]:
            batch["entity_PILs"] = [example["entity_PILs"] for example in examples]
        batch["targets"] = [example["targets"] for example in examples]

        return batch
