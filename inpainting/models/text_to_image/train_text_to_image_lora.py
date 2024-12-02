#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fine-tuning script for Stable Diffusion for text2image with support for LoRA."""

import argparse
import logging
import math
import os
import random
import shutil
import yaml
from contextlib import nullcontext
from pathlib import Path

import click
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from functools import partial
from hydra.utils import instantiate

import diffusers
from diffusers import (
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, compute_snr
from diffusers.utils import (
    convert_state_dict_to_diffusers,
    is_wandb_available,
)
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from ...data.datasets import InpaintingDataset


if is_wandb_available():
    import wandb


logger = get_logger(__name__, log_level="INFO")


@click.group()
def cli():
    pass


def save_model_card(
    repo_id: str,
    images: list = None,
    base_model: str = None,
    dataset_name: str = None,
    repo_folder: str = None,
):
    img_str = ""
    if images is not None:
        for i, image in enumerate(images):
            image.save(os.path.join(repo_folder, f"image_{i}.png"))
            img_str += f"![img_{i}](./image_{i}.png)\n"

    model_description = f"""
    # LoRA text2image fine-tuning - {repo_id}
    These are LoRA adaption weights for {base_model}. The weights were fine-tuned on the {dataset_name} dataset. You can find some example images in the following. \n
    {img_str}
    """

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="creativeml-openrail-m",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "stable-diffusion",
        "stable-diffusion-diffusers",
        "text-to-image",
        "diffusers",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def log_validation(
    pipeline,
    args,
    accelerator,
    epoch,
    is_final_validation=False,
):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)
    generator = torch.Generator(device=accelerator.device)
    if args.seed is not None:
        generator = generator.manual_seed(args.seed)
    images = []
    if torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type)

    with autocast_ctx:
        for _ in range(args.num_validation_images):
            images.append(
                pipeline(
                    args.validation_prompt, num_inference_steps=30, generator=generator
                ).images[0]
            )

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    phase_name: [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                        for i, image in enumerate(images)
                    ]
                }
            )
    return images


@cli.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    help="The path to a YAML configuration file.",
)
def main(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = instantiate(config_dict)

    # Define the Accelerator
    accelerator: Accelerator = config.accelerator

    # Disable AMP for MPS.
    if torch.backends.mps.is_available():
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Set the seed
    assert "seed" in config and isinstance(config.seed, int), "Seed must be an integer."
    set_seed(config.seed + config.accelerator.process_index)

    # Load scheduler, tokenizer and models.
    noise_scheduler = config.noise_scheduler
    tokenizer = config.tokenizer
    text_encoder = config.text_encoder
    vae = config.vae
    unet = config.unet

    # Freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # For mixed precision training we cast all non-trainable weights (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Freeze the unet parameters before adding adapters
    for param in unet.parameters():
        param.requires_grad_(False)

    # Load the LoRA config
    unet_lora_config = config.model.lora_config_partial(
        target_modules=["to_k", "to_q", "to_v", "to_out.0"]
    )

    # Move unet, vae and text_encoder to device and cast to weight_dtype
    unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Add adapter and make sure the trainable params are in float32.
    unet.add_adapter(unet_lora_config)
    if accelerator.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    if config.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warning(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError(
                "xformers is not available. Make sure it is installed correctly"
            )

    # Get the parameters that require gradients
    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())

    # Enable gradient checkpointing if needed
    if config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Assignments
    checkpointing_steps = config.checkpointing_steps
    do_early_stopping = config.do_early_stopping
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    learning_rate = config.learning_rate
    max_grad_norm = config.max_grad_norm
    max_train_samples = config.max_train_samples
    max_train_steps = config.max_train_steps
    noise_offset = config.noise_offset
    num_train_epochs = config.num_train_epochs
    output_dir = accelerator.project_configuration.project_dir
    resolution = config.resolution
    resume_from_checkpoint = config.resume_from_checkpoint
    snr_gamma = config.snr_gamma
    train_batch_size = config.batch_size

    # Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size
    if config.scale_lr:
        learning_rate = (
            learning_rate
            * gradient_accumulation_steps
            * train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = config.optimizer_partial(
        lora_layers,
        lr=learning_rate,
    )

    # Define the datasets
    train_dataset = config.datasets.train_partial(tokenizer=tokenizer)
    val_dataset = config.datasets.val_partial(tokenizer=tokenizer)

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # If training with limited samples
    with accelerator.main_process_first():
        if max_train_samples is not None:
            random.shuffle(train_dataset.image_paths)
            train_dataset.image_paths = train_dataset.image_paths[:max_train_samples]
            train_dataset.image_stems = [
                image_path.stem for image_path in train_dataset.image_paths
            ]

    # Define the collate_fn
    collate_fn = InpaintingDataset.collate_fn

    # DataLoaders creation
    train_dataloader = config.dataloaders.train_partial(
        train_dataset,
        collate_fn=collate_fn,
    )
    val_dataloader = config.dataloaders.val_partial(
        val_dataset,
        collate_fn=collate_fn,
        worker_init_fn=lambda worker_id: val_dataset.generator.manual_seed(
            config.seed + accelerator.process_index + worker_id
        ),
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(
            len(train_dataloader) / accelerator.num_processes
        )
        num_update_steps_per_epoch = math.ceil(
            len_train_dataloader_after_sharding / gradient_accumulation_steps
        )
        num_training_steps_for_scheduler = (
            num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = max_train_steps * accelerator.num_processes
    num_warmup_steps_for_scheduler = (
        num_update_steps_per_epoch * config.scheduler.warmup_ratio
    )

    lr_scheduler = get_scheduler(
        config.scheduler.name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps_for_scheduler,
        num_training_steps=num_training_steps_for_scheduler,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if max_train_steps is None:
        max_train_steps = num_train_epochs * num_update_steps_per_epoch
        if (
            num_training_steps_for_scheduler
            != max_train_steps * accelerator.num_processes
        ):
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        if accelerator.log_with:
            tracker_name = accelerator.log_with[0].value
        accelerator.init_trackers(
            "WhatAndWhere",
            init_kwargs={tracker_name: config.init_trackers_kwargs},
            config=config_dict,
        )

    # Train!
    total_batch_size = (
        train_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )
    val_progress_bar = tqdm(
        range(0, len(val_dataloader)),
        desc="Validation steps",
        disable=not accelerator.is_local_main_process,
    )
    val_step = 0
    val_last_avg_loss = float("inf")

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                loss = forward(
                    batch,
                    vae,
                    unet,
                    text_encoder,
                    noise_scheduler,
                    config,
                    weight_dtype,
                    resolution,
                    noise_offset,
                    snr_gamma,
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    params_to_clip = lora_layers
                    accelerator.clip_grad_norm_(params_to_clip, max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # Logging!
                accelerator.log(
                    {
                        "train/loss": train_loss,
                        "train/lr": lr_scheduler.get_last_lr()[0],
                        "train/step": global_step,
                    }
                )
                train_loss = 0.0

                # Save the model and the lora layers
                if (
                    checkpointing_steps is not None
                    and global_step % checkpointing_steps == 0
                ):
                    if accelerator.is_main_process:
                        save_path = os.path.join(
                            output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)

                        unwrapped_unet = unwrap_model(unet)
                        unet_lora_state_dict = convert_state_dict_to_diffusers(
                            get_peft_model_state_dict(unwrapped_unet)
                        )

                        StableDiffusionPipeline.save_lora_weights(
                            save_directory=save_path,
                            unet_lora_layers=unet_lora_state_dict,
                            safe_serialization=True,
                        )

                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

        ### VALIDATION
        val_noise_generator = torch.Generator(device=accelerator.device).manual_seed(
            config.seed + accelerator.process_index
        )
        unet.eval()
        avg_val_loss = 0.0
        val_loss = 0.0
        for step, batch in enumerate(val_dataloader):
            with torch.no_grad():
                loss = forward(
                    batch,
                    vae,
                    unet,
                    text_encoder,
                    noise_scheduler,
                    config,
                    weight_dtype,
                    resolution,
                    noise_offset,
                    snr_gamma,
                    generator=val_noise_generator,
                )

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                val_loss += avg_loss.item()
                avg_val_loss += val_loss
                val_step += 1

            logs = {"val/step_loss": val_loss, "val/step": val_step}
            accelerator.log(logs)
            val_loss = 0.0
            val_progress_bar.update(1)
            val_progress_bar.set_postfix(**logs)

        avg_val_loss /= len(val_dataloader)
        accelerator.log({"val/avg_loss": avg_val_loss, "val/step": val_step})

        # Save the lora layers
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unet = unet.to(torch.float32)

            unwrapped_unet = unwrap_model(unet)
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unwrapped_unet)
            )
            StableDiffusionPipeline.save_lora_weights(
                save_directory=os.path.join(output_dir, f"checkpoint-{global_step}"),
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )

        # Do early stopping if needed
        if do_early_stopping:
            if avg_val_loss > val_last_avg_loss:
                logger.info(
                    f"Early stopping at epoch {epoch} with validation loss {avg_val_loss}."
                )
                break
            val_last_avg_loss = avg_val_loss

    accelerator.end_training()


def forward(
    batch,
    vae,
    unet,
    text_encoder,
    noise_scheduler,
    config,
    weight_dtype,
    resolution,
    noise_offset,
    snr_gamma,
    generator=None,
):
    # Convert images to latent space
    latents = vae.encode(
        batch["pixel_values"].to(dtype=weight_dtype)
    ).latent_dist.sample()
    latents = latents * vae.config.scaling_factor

    # Convert masked images to latent space
    masked_latents = vae.encode(
        batch["masked_pixel_values"].to(dtype=weight_dtype)
    ).latent_dist.sample()
    masked_latents = masked_latents * vae.config.scaling_factor

    # Resize the mask to latents shape as we concatenate the mask to the latents
    masks = torch.nn.functional.interpolate(
        batch["masks"], size=(resolution // 8, resolution // 8)
    )

    # Sample noise that we'll add to the latents
    noise = torch.empty_like(latents).normal_(generator=generator)
    if noise_offset:
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += noise_offset * torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1),
            device=latents.device,
        )
    bsz = latents.shape[0]

    # Sample a random timestep for each image
    timesteps = torch.randint(
        0,
        noise_scheduler.config.num_train_timesteps,
        (bsz,),
        device=latents.device,
        generator=generator,
    )
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    # (this is the forward diffusion process)
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    # Concatenate the noisy latents with the masks and the masked latents
    latent_model_input = torch.cat([noisy_latents, masks, masked_latents], dim=1).to(
        dtype=weight_dtype
    )

    # Get the text embedding for conditioning
    encoder_hidden_states = text_encoder(batch["input_ids"], return_dict=False)[0]

    # Get the target for loss depending on the prediction type
    if config.prediction_type is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=config.prediction_type)

    # Define the target
    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(
            f"Unknown prediction type {noise_scheduler.config.prediction_type}"
        )
    
    # Freestyle inputs
    freestyle_inputs = {}
    if config.freestyle:
        freestyle_inputs["freestyle_attention_mask"] = batch["freestyle_attention_mask"]
        freestyle_inputs["freestyle_layout_mask"] = batch["freestyle_layout_mask"]

    # Predict the noise residual and compute loss
    model_pred = unet(
        latent_model_input,
        timesteps,
        encoder_hidden_states,
        return_dict=False,
        **freestyle_inputs,
    )[0]

    # Compute the loss
    if snr_gamma is None:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack(
            [snr, snr_gamma * torch.ones_like(timesteps)], dim=1
        ).min(dim=1)[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
        loss = loss.mean()

    return loss


if __name__ == "__main__":
    cli()
