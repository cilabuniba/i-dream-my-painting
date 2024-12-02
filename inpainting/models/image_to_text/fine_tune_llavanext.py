import math
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.logging import get_logger
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)
from torchmetrics import MeanMetric
from torchmetrics.text import BLEUScore, ROUGEScore, BERTScore
from ..metrics import CLIPScoreText2Image
from PIL import Image
from transformers import (
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)
import click
import yaml
import os
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
    get_scheduler,
)
from hydra.utils import instantiate
from .schedulers import AlphaScheduler
from .samplers import AlphaScheduleBatchSampler
import timeit
from wandb import Image as WandbImage
from safetensors.torch import save_file

from ...data.datasets import LlavaDataset
from .schedulers import CurriculumAlphaScheduler

logger = get_logger(__name__, log_level="INFO")


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config-path",
    type=click.Path(exists=True),
    help="The path to a YAML configuration file.",
)
def train(
    config_path,
):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    config = instantiate(config_dict)

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    # Set the seed
    assert "seed" in config and isinstance(config.seed, int), "Seed must be an integer."
    set_seed(config.seed + config.accelerator.process_index)

    # Load the processor
    processor = AutoProcessor.from_pretrained(config.model.id)
    processor.tokenizer.padding_side = (
        "right"  # during training, one always uses padding on the right
    )

    # Accelerator
    accelerator: Accelerator = config.accelerator

    # For checkpoints
    resume_from_checkpoint = config.resume_from_checkpoint
    output_dir = accelerator.project_configuration.project_dir

    # Load the model
    fine_tuning_type = config.model.fine_tuning_type
    assert fine_tuning_type in [
        "full",
        "quantized",
        "lora",
        "qlora",
    ], f"Fine-tuning type '{fine_tuning_type}' not recognized."
    if fine_tuning_type in ["lora", "qlora"]:
        if fine_tuning_type == "qlora":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = LlavaNextForConditionalGeneration.from_pretrained(
            config.model.id,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
    elif fine_tuning_type == "full":
        # for full fine-tuning, we can speed up the model using Flash Attention
        # only available on certain devices, see https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
        model = LlavaNextForConditionalGeneration.from_pretrained(
            config.model.id,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
        )
    elif fine_tuning_type == "quantized":
        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = LlavaNextForConditionalGeneration.from_pretrained(config.model.id, quantization_config=quantization_config, device_map="auto")


    checkpoint = None
    if fine_tuning_type in ["lora", "qlora"]:
        # Configure LoRA
        def find_all_linear_names(model):
            cls = torch.nn.Linear
            lora_module_names = []
            for name, module in model.named_modules():
                if not "language_model" in name:
                    continue
                if "lm_head" in name:
                    continue
                if isinstance(module, cls):
                    lora_module_names.append(name)
            return lora_module_names

        lora_config: LoraConfig = config.model.lora_config_partial(
            target_modules=find_all_linear_names(model),
        )

        # Prepare the model for training with LoRA
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

        # Get the LoRA layers
        if resume_from_checkpoint is not None:
            if resume_from_checkpoint == "latest":
                # Get the most recent checkpoint
                dirs = os.listdir(output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                checkpoint = dirs[-1] if len(dirs) > 0 else None
            else:
                checkpoint = os.path.basename(resume_from_checkpoint)
            model = PeftModel.from_pretrained(
                model, os.path.join(output_dir, checkpoint, "adapter"), is_trainable=True
            )
        else:
            model = get_peft_model(model, lora_config)

    # Define layers to optimize
    if fine_tuning_type in ["lora", "qlora"]:
        layers_to_optimize = filter(lambda p: p.requires_grad, model.parameters())
    else:
        layers_to_optimize = model.parameters()

    # Load the alpha scheduler
    train_alpha_scheduler: AlphaScheduler = config.alpha_schedulers.train
    val_alpha_scheduler: AlphaScheduler = config.alpha_schedulers.val

    # Load the datasets
    train_dataset: LlavaDataset = config.datasets.train
    val_dataset: LlavaDataset = config.datasets.val

    # Load the batch samplers (they are responsible for the update of the alpha values)
    train_sampler: AlphaScheduleBatchSampler = config.batch_samplers.train_partial(
        dataset=train_dataset,
        alpha_scheduler=train_alpha_scheduler,
        num_processes=accelerator.num_processes,
        gradient_accumulation_steps=accelerator.gradient_accumulation_steps,
    )
    val_sampler: AlphaScheduleBatchSampler = config.batch_samplers.val_partial(
        dataset=val_dataset,
        alpha_scheduler=val_alpha_scheduler,
        num_processes=accelerator.num_processes,
        gradient_accumulation_steps=accelerator.gradient_accumulation_steps,
    )

    # Get the dataloaders
    train_dataloader = config.dataloaders.train_partial(
        dataset=train_dataset,
        batch_sampler=train_sampler,
        collate_fn=partial(
            LlavaDataset.collate_fn,
            processor=processor,
            max_length=config.max_length,
            split="train",
        ),
    )
    val_overwrite_prompt = None
    if fine_tuning_type == "quantized":
        val_overwrite_prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image>\nWrite 1 text prompt that describes a reasonable object to be inserted in the gray area. ASSISTANT:"
    val_dataloader = config.dataloaders.val_partial(
        dataset=val_dataset,
        batch_sampler=val_sampler,
        collate_fn=partial(
            LlavaDataset.collate_fn,
            processor=processor,
            max_length=config.max_length,
            overwrite_prompt=val_overwrite_prompt,
            split="val",
        ),
        worker_init_fn=lambda worker_id: val_dataset.generator.manual_seed(
            config.seed + accelerator.process_index + worker_id
        ),  # set the seed for the dataset
    )

    # Initialize the optimizer
    optimizer = config.optimizer_partial(
        params=layers_to_optimize,
    )

    # Useful assignments
    batch_size = train_sampler.batch_size
    checkpointing_steps = config.checkpointing_steps
    gradient_accumulation_steps = accelerator.gradient_accumulation_steps
    max_grad_norm = config.max_grad_norm
    max_length = config.max_length
    max_train_steps = config.max_train_steps
    max_val_steps = (
        math.ceil(config.max_val_instances / val_sampler.batch_size) // accelerator.num_processes
    )
    num_train_epochs = config.num_train_epochs
    num_train_images_to_log = config.num_train_images_to_log
    num_val_images_to_log = config.num_val_images_to_log
    only_val = config.only_val
    validate_every_n_epochs = config.validate_every_n_epochs

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
        num_update_steps_per_epoch * accelerator.num_processes * config.scheduler.warmup_ratio
    )
    lr_scheduler = get_scheduler(
        name=config.scheduler.name,
        optimizer=optimizer,
        num_training_steps=num_training_steps_for_scheduler,
        num_warmup_steps=num_warmup_steps_for_scheduler,
    )

    # Prepare!
    model, optimizer, train_dataloader, val_dataloader, lr_scheduler = (
        accelerator.prepare(
            model, optimizer, train_dataloader, val_dataloader, lr_scheduler
        )
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

    # We need to initialize the trackers we use, and also store our configuration
    tracker_name = None
    if accelerator.is_main_process:
        if accelerator.log_with:
            tracker_name = accelerator.log_with[0].value
        accelerator.init_trackers(
            "WhatAndWhere",
            init_kwargs={tracker_name: config.init_trackers_kwargs},
            config=config_dict,
        )

    # Define the total batch size
    total_batch_size = (
        batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    # Print stuff
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  Total warmup steps = {num_warmup_steps_for_scheduler}")
    global_step = 0
    first_epoch = 0

    # We need to custom save and load models
    def save_model_hook(models, weights, output_dir):
        while len(weights) > 0:
            weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            models.pop()

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Potentially load in the weights and states from a previous save
    if checkpoint is not None and not only_val:
        accelerator.print(f"Resuming from checkpoint {checkpoint}")
        accelerator.load_state(os.path.join(output_dir, checkpoint))
        global_step = int(checkpoint.split("-")[1])

        initial_global_step = global_step
        first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    # Setup the alpha scheduler
    train_alpha_scheduler.set_num_steps(
        math.ceil(max_train_steps - (0.5 * max_train_steps))
    )
    train_alpha_scheduler.set_step(global_step)

    # Setup the progress bar
    progress_bar = tqdm(
        range(0, max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    # Define the validation metrics
    clip_model_id = "openai/clip-vit-base-patch32"
    accuracy = MeanMetric().to(accelerator.device)
    accuracy_last = MeanMetric().to(accelerator.device)
    bleu1 = BLEUScore(n_gram=1).to(accelerator.device)
    bleu1_last = BLEUScore(n_gram=1).to(accelerator.device)
    bleu4 = BLEUScore(n_gram=4).to(accelerator.device)
    bleu4_last = BLEUScore(n_gram=4).to(accelerator.device)
    rouge = ROUGEScore().to(accelerator.device)
    rouge_last = ROUGEScore().to(accelerator.device)
    bert_score = BERTScore(device=accelerator.device).to(accelerator.device)
    bert_score_last = BERTScore(device=accelerator.device).to(accelerator.device)
    clip_text_model = CLIPTextModelWithProjection.from_pretrained(
        clip_model_id, local_files_only=True, use_safetensors=False
    ).to(accelerator.device)
    clip_vision_model = CLIPVisionModelWithProjection.from_pretrained(
        clip_model_id, local_files_only=True, use_safetensors=False
    ).to(accelerator.device)
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id)
    clip_score = CLIPScoreText2Image(
        text_model=clip_text_model,
        vision_model=clip_vision_model,
        processor=clip_processor,
    ).to(accelerator.device)
    clip_score_last = CLIPScoreText2Image(
        text_model=clip_text_model,
        vision_model=clip_vision_model,
        processor=clip_processor,
    ).to(accelerator.device)

    # Training loop
    for epoch in range(first_epoch, num_train_epochs):
        if not only_val:
            ### Training ###
            model.train()
            processor.tokenizer.padding_side = (
                "right"  # during training, one always uses padding on the right
            )
            train_loss = 0.0
            train_images_to_log = []
            for step, batch in enumerate(train_dataloader):
                with accelerator.accumulate(model):
                    outputs = model(**batch["model_inputs"])
                    loss = outputs.loss

                    # Gather the losses across all processes for logging (if we use distributed training).
                    avg_loss = accelerator.gather(loss.repeat(batch_size)).mean()
                    if not torch.isfinite(avg_loss):
                        # skip steps with bad loss
                        raise RuntimeError("Loss is not finite. Stopping training.")

                    train_loss += avg_loss.item() / gradient_accumulation_steps

                    # Backpropagate
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        params_to_clip = layers_to_optimize
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
                        },
                        step=global_step,
                    )
                    train_loss = 0.0

                    # Log images
                    if accelerator.is_main_process:
                        if (
                            tracker_name == "wandb"
                            and len(train_images_to_log) < num_train_images_to_log
                            and global_step
                            % (num_update_steps_per_epoch // num_train_images_to_log)
                            == 0
                        ):
                            image, masked_image = (
                                batch["PIL_images"][0],
                                batch["PIL_masked_images"][0],
                            )
                            width, height = image.size
                            new_image = Image.new("RGB", (width, height * 2))
                            new_image.paste(image, (0, 0))
                            new_image.paste(masked_image, (0, height))
                            train_images_to_log.append(
                                WandbImage(
                                    new_image,
                                    caption=f"Epoch: {epoch} Step: {global_step} Target: {batch['targets'][0]}",
                                )
                            )

                    # Checkpointing!
                    if (
                        checkpointing_steps is not None
                        and global_step % config.checkpointing_steps == 0
                    ):
                        if accelerator.is_main_process:
                            save_path = os.path.join(
                                output_dir, f"checkpoint-{global_step}"
                            )
                            accelerator.save_state(save_path)
                            unwrapped_model = accelerator.unwrap_model(model)
                            unwrapped_model.save_pretrained(
                                os.path.join(save_path, "adapter"),
                                save_function=accelerator.save,
                            )
                            logger.info(f"Saving checkpoint to {save_path}")

                # Update the progress bar
                logs = {
                    "steploss": loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                }
                progress_bar.set_postfix(**logs)

                # Exit the loop if we reach the maximum number of training steps
                if global_step >= max_train_steps:
                    break

            # Log the images
            if tracker_name == "wandb" and len(train_images_to_log) > 0:
                accelerator.log(
                    {"train/images": train_images_to_log, "train/step": global_step},
                    step=global_step,
                )

            # Save the epoch's checkpoint
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    os.path.join(save_path, "adapter"),
                    save_function=accelerator.save,
                )
                logger.info(f"Saving checkpoint to {save_path}")

        ### Validation ###
        accelerator.wait_for_everyone()
        if (epoch + 1) % validate_every_n_epochs == 0:
            val_progress_bar = tqdm(
                range(0, len(val_dataloader)),
                initial=0,
                desc="Val Steps",
                # Only show the progress bar once on each machine.
                disable=not accelerator.is_local_main_process,
            )
            model.eval()
            processor.tokenizer.padding_side = (
                "left"  # during training, one always uses padding on the right
            )
            logger.info("Validation...")
            val_images_to_log = []
            for step, batch in enumerate(val_dataloader):
                if val_dataset.only_gray_concept:
                    stop_strings = ["</gray>"]
                elif val_overwrite_prompt is not None:
                    stop_strings = ["."]
                else:
                    stop_strings = ["</red>"]
                with torch.no_grad():
                    generated_ids = accelerator.unwrap_model(model).generate(
                        **batch["model_inputs"],
                        max_new_tokens=max_length,
                        stop_strings=stop_strings,
                        tokenizer=processor.tokenizer,
                    )
                    predictions = processor.batch_decode(
                        generated_ids,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )

                    clean_predictions = []
                    for prediction in predictions:
                        keyword = "ASSISTANT: "
                        keyword_idx = prediction.find(keyword)
                        clean_predictions.append(
                            prediction[keyword_idx + len(keyword) :].strip().lower()
                        )

                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if not os.path.exists(os.path.join(output_dir, "predictions.txt")):
                        # create empty file
                        with open(os.path.join(output_dir, "predictions.txt"), "w") as f:
                            pass
                    with open(os.path.join(output_dir, "predictions.txt"), "a") as f:
                        for i, prediction in enumerate(clean_predictions):
                            f.write(f"{i}\t{prediction}\n")

                    if val_overwrite_prompt is None:
                        color_predictions = []
                        for i, prediction in enumerate(clean_predictions):
                            instance_color_predictions = []
                            for color in batch["entity_colors"][i]:
                                if f"<{color}>" in prediction and f"</{color}>" in prediction:
                                    color_start = prediction.find(f"<{color}>")
                                    color_end = prediction.find(f"</{color}>")
                                    instance_color_predictions.append(
                                        prediction[color_start + len(f"<{color}>") : color_end]
                                    )
                                else:
                                    instance_color_predictions.append("")
                            color_predictions.append(instance_color_predictions)
                    else:
                        color_predictions = [[clean_prediction] for clean_prediction in clean_predictions]
                        
                    # Remove nested lists
                    color_predictions = [
                        item for sublist in color_predictions for item in sublist
                    ]
                    noun_chunk_roots = [
                        item
                        for sublist in batch["entity_noun_chunk_roots"]
                        for item in sublist
                    ]
                    references = [
                        [item] for sublist in batch["entity_captions"] for item in sublist
                    ]
                    references_for_bert_score = [
                        item for sublist in batch["entity_captions"] for item in sublist
                    ]
                    PILs = [item for sublist in batch["entity_PILs"] for item in sublist]

                    # Compute accuracy
                    noun_chunk_roots_predictions = []
                    for prediction, noun_chunk_root in zip(
                        color_predictions, noun_chunk_roots
                    ):
                        if noun_chunk_root in prediction:
                            noun_chunk_roots_predictions.append(1)
                        else:
                            noun_chunk_roots_predictions.append(0)

                    # Get the last predictions
                    n_colors_per_instance = [len(colors) for colors in batch["entity_colors"]]
                    color_predictions_last = []
                    references_last = []
                    references_for_bert_score_last = []
                    PILs_last = []
                    noun_chunk_roots_predictions_last = []
                    tot_colors_seen = 0
                    for i, n_colors in enumerate(n_colors_per_instance):
                        color_predictions_last.append(color_predictions[tot_colors_seen + n_colors - 1])
                        references_last.append(references[tot_colors_seen + n_colors - 1])
                        references_for_bert_score_last.append(references_for_bert_score[tot_colors_seen + n_colors - 1])
                        PILs_last.append(PILs[tot_colors_seen + n_colors - 1])
                        noun_chunk_roots_predictions_last.append(
                            noun_chunk_roots_predictions[tot_colors_seen + n_colors - 1]
                        )
                        tot_colors_seen += n_colors

                    # Compute metrics
                    batch_accuracy = accuracy(torch.tensor(noun_chunk_roots_predictions).to(accelerator.device))
                    batch_accuracy_last = accuracy_last(torch.tensor(noun_chunk_roots_predictions_last).to(accelerator.device))
                    batch_bleu1 = bleu1(color_predictions, references)
                    batch_bleu1_last = bleu1_last(color_predictions_last, references_last)
                    batch_bleu4 = bleu4(color_predictions, references)
                    batch_bleu4_last = bleu4_last(color_predictions_last, references_last)
                    rouge.update(color_predictions, references)  # only compute at the end
                    rouge_last.update(color_predictions_last, references_last)  # only compute at the end
                    bert_score.update(
                        color_predictions, references_for_bert_score
                    )  # only compute at the end
                    bert_score_last.update(
                        color_predictions_last, references_for_bert_score_last
                    )
                    batch_clip_score = clip_score(
                        texts=[
                            f"An image of {prediction}" for prediction in color_predictions
                        ],
                        images=PILs,
                    )
                    batch_clip_score_last = clip_score_last(
                        texts=[
                            f"An image of {prediction}" for prediction in color_predictions_last
                        ],
                        images=PILs_last,
                    )

                    # Prepare the log for the images
                    if accelerator.is_main_process:
                        if (
                            tracker_name == "wandb"
                            and len(val_images_to_log) < num_val_images_to_log
                        ):
                            for i, (image, masked_image) in enumerate(
                                zip(batch["PIL_images"], batch["PIL_masked_images"])
                            ):
                                width, height = image.size
                                new_image = Image.new("RGB", (width, height * 2))
                                new_image.paste(image, (0, 0))
                                new_image.paste(masked_image, (0, height))
                                val_images_to_log.append(
                                    WandbImage(
                                        new_image,
                                        caption=f"Target: {batch['targets'][i]}\nPrediction: {clean_predictions[i]}",
                                    )
                                )
                                if len(val_images_to_log) >= num_val_images_to_log:
                                    break

                    # Exit the loop if we reach the maximum number of validation steps
                    if step + 1 >= max_val_steps:
                        break

                if accelerator.is_local_main_process:
                    val_progress_bar.update(1)
                    logs = {
                        "acc": batch_accuracy.item(),
                        "bleu1": batch_bleu1.item(),
                        "bleu4": batch_bleu4.item(),
                        "clipscore": batch_clip_score.item(),
                    }
                    val_progress_bar.set_postfix(**logs)

            accelerator.wait_for_everyone()

            # Log the validation metrics
            computed_rouge = rouge.compute()
            computed_rouge_last = rouge_last.compute()
            computed_bert_score = bert_score.compute()
            computed_bert_score_last = bert_score_last.compute()
            computed_rouge = {f"val/{key}": value for key, value in computed_rouge.items()}
            computed_rouge_last = {f"val/{key}_last": value for key, value in computed_rouge_last.items()}
            computed_bert_score = {
                f"val/bertscore_{key}": value.mean()
                for key, value in computed_bert_score.items()
            }
            computed_bert_score_last = {
                f"val/bertscore_{key}_last": value.mean()
                for key, value in computed_bert_score_last.items()
            }
            computed_accuracy = accuracy.compute()
            computed_accuracy_last = accuracy_last.compute()
            computed_bleu1 = bleu1.compute()
            computed_bleu1_last = bleu1_last.compute()
            computed_bleu4 = bleu4.compute()
            computed_bleu4_last = bleu4_last.compute()
            computed_clip_score = clip_score.compute()
            computed_clip_score_last = clip_score_last.compute()

            computed_metrics = {
                "val/accuracy": computed_accuracy,
                "val/accuracy_last": computed_accuracy_last,
                "val/bleu1": computed_bleu1,
                "val/bleu1_last": computed_bleu1_last,
                "val/bleu4": computed_bleu4,
                "val/bleu4_last": computed_bleu4_last,
                "val/clipscore": computed_clip_score,
                "val/clipscore_last": computed_clip_score_last,
                "val/epoch": epoch,
            }
            computed_metrics.update(computed_rouge)
            computed_metrics.update(computed_rouge_last)
            computed_metrics.update(computed_bert_score)
            computed_metrics.update(computed_bert_score_last)
            accelerator.log(computed_metrics, step=global_step)

            # Log the images
            if ( accelerator.is_main_process and tracker_name == "wandb"
                and len(val_images_to_log) > 0
            ):
                accelerator.log(
                    {"val/images": val_images_to_log, "val/epoch": epoch}, step=global_step
                )

            # Reset the metrics
            accuracy.reset()
            accuracy_last.reset()
            bleu1.reset()
            bleu1_last.reset()
            bleu4.reset()
            bleu4_last.reset()
            rouge.reset()
            rouge_last.reset()
            bert_score.reset()
            bert_score_last.reset()
            clip_score.reset()
            clip_score_last.reset()

        # Empty cache
        torch.cuda.empty_cache()

        # Exit the loop if only validation
        if only_val:
            break

    accelerator.end_training()


if __name__ == "__main__":
    cli()
