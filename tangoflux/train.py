import time
import argparse
import json
import logging
import math
import os
import yaml
from pathlib import Path
import diffusers
import datasets
import numpy as np
import pandas as pd
import wandb
import transformers
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from transformers import SchedulerType, get_scheduler
from model import TangoFlux
from datasets import load_dataset, Audio
from utils import Text2AudioDataset, read_wav_file, pad_wav

from diffusers import AutoencoderOobleck
import torchaudio

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rectified flow for text to audio generation task."
    )

    parser.add_argument(
        "--num_examples",
        type=int,
        default=-1,
        help="How many examples to use for training and validation.",
    )

    parser.add_argument(
        "--text_column",
        type=str,
        default="captions",
        help="The name of the column in the datasets containing the input texts.",
    )
    parser.add_argument(
        "--audio_column",
        type=str,
        default="location",
        help="The name of the column in the datasets containing the audio paths.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.95,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="tangoflux_config.yaml",
        help="Config file defining the model size as well as other hyper parameter.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",
        help="Add prefix in text prompts.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay to use."
    )

    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=1e-2,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="best",
        help="Whether the various states should be saved at the end of every 'epoch' or 'best' whenever validation loss decreases.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=5,
        help="Save model after every how many epochs when checkpointing_steps is set to best.",
    )

    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="If the training should continue from a local checkpoint folder.",
    )

    parser.add_argument(
        "--load_from_checkpoint",
        type=str,
        default=None,
        help="Whether to continue training from a model weight",
    )

    parser.add_argument(
        "--load_from_hf",
        action='store_true',
        default=False,
        help="Whether to Finetune a HuggingFace checkpoint",
    )

    parser.add_argument(
        "--hf_repo",
        type=str,
        default="declare-lab/TangoFlux",
        help="HuggingFace repo to finetune",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    accelerator_log_kwargs = {}

    def load_config(config_path):
        with open(config_path, "r") as file:
            return yaml.safe_load(file)

    config = load_config(args.config)

    learning_rate = float(config["training"]["learning_rate"])
    num_train_epochs = int(config["training"]["num_train_epochs"])
    num_warmup_steps = int(config["training"]["num_warmup_steps"])
    per_device_batch_size = int(config["training"]["per_device_batch_size"])
    gradient_accumulation_steps = int(config["training"]["gradient_accumulation_steps"])

    output_dir = config["paths"]["output_dir"]

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        **accelerator_log_kwargs,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    datasets.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory creation and wandb tracking
    if accelerator.is_main_process:
        if output_dir is None or output_dir == "":
            output_dir = "saved/" + str(int(time.time()))

            if not os.path.exists("saved"):
                os.makedirs("saved")

            os.makedirs(output_dir, exist_ok=True)

        elif output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)

        os.makedirs("{}/{}".format(output_dir, "outputs"), exist_ok=True)
        with open("{}/summary.jsonl".format(output_dir), "a") as f:
            f.write(json.dumps(dict(vars(args))) + "\n\n")

        accelerator.project_configuration.automatic_checkpoint_naming = False

        wandb.init(
            project="Text to Audio Flow matching",
            settings=wandb.Settings(_disable_stats=True),
        )

    accelerator.wait_for_everyone()

    # Get the datasets
    data_files = {}
    # if args.train_file is not None:
    if config["paths"]["train_file"] != "":
        data_files["train"] = config["paths"]["train_file"]
    # if args.validation_file is not None:
    if config["paths"]["val_file"] != "":
        data_files["validation"] = config["paths"]["val_file"]
    if config["paths"]["test_file"] != "":
        data_files["test"] = config["paths"]["test_file"]
    else:
        data_files["test"] = config["paths"]["val_file"]

    extension = "json"
    raw_datasets = load_dataset(extension, data_files=data_files)
    text_column, audio_column = args.text_column, args.audio_column

    model = TangoFlux(config=config["model"])
    vae = AutoencoderOobleck.from_pretrained(
        "stabilityai/stable-audio-open-1.0", subfolder="vae"
    )

    ## Freeze vae
    for param in vae.parameters():
        vae.requires_grad = False
        vae.eval()

    ## Freeze text encoder param
    for param in model.text_encoder.parameters():
        param.requires_grad = False
        model.text_encoder.eval()

    prefix = args.prefix

    with accelerator.main_process_first():
        train_dataset = Text2AudioDataset(
            raw_datasets["train"],
            prefix,
            text_column,
            audio_column,
            "duration",
            args.num_examples,
        )
        eval_dataset = Text2AudioDataset(
            raw_datasets["validation"],
            prefix,
            text_column,
            audio_column,
            "duration",
            args.num_examples,
        )
        test_dataset = Text2AudioDataset(
            raw_datasets["test"],
            prefix,
            text_column,
            audio_column,
            "duration",
            args.num_examples,
        )

        accelerator.print(
            "Num instances in train: {}, validation: {}, test: {}".format(
                train_dataset.get_num_instances(),
                eval_dataset.get_num_instances(),
                test_dataset.get_num_instances(),
            )
        )

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config["training"]["per_device_batch_size"],
        collate_fn=train_dataset.collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        shuffle=True,
        batch_size=config["training"]["per_device_batch_size"],
        collate_fn=eval_dataset.collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config["training"]["per_device_batch_size"],
        collate_fn=test_dataset.collate_fn,
    )

    # Optimizer

    optimizer_parameters = list(model.transformer.parameters()) + list(
        model.fc.parameters()
    )
    num_trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    accelerator.print("Num trainable parameters: {}".format(num_trainable_parameters))

    if args.load_from_checkpoint:
        from safetensors.torch import load_file

        w1 = load_file(args.load_from_checkpoint)
        model.load_state_dict(w1, strict=False)
        logger.info("Weights loaded from{}".format(args.load_from_checkpoint))
    elif args.load_from_hf:
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download

        path = snapshot_download(repo_id=args.hf_repo)
        w1 = load_file(f"{path}/tangoflux.safetensors")
        model.load_state_dict(w1, strict=False)
        logger.info(f"Weights loaded from {args.hf_repo}")

    optimizer = torch.optim.AdamW(
        optimizer_parameters,
        lr=learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps
        * gradient_accumulation_steps
        * accelerator.num_processes,
        num_training_steps=args.max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    vae, model, optimizer, lr_scheduler = accelerator.prepare(
        vae, model, optimizer, lr_scheduler
    )

    train_dataloader, eval_dataloader, test_dataloader = accelerator.prepare(
        train_dataloader, eval_dataloader, test_dataloader
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.

    # Train!
    total_batch_size = (
        per_device_batch_size * accelerator.num_processes * gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {per_device_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )

    completed_steps = 0
    starting_epoch = 0
    # Potentially load in the weights and states from a previous save
    resume_from_checkpoint = config["paths"]["resume_from_checkpoint"]
    if resume_from_checkpoint != "":
        accelerator.load_state(resume_from_checkpoint)
        accelerator.print(f"Resumed from local checkpoint: {resume_from_checkpoint}")

    # Duration of the audio clips in seconds
    best_loss = np.inf
    length = config["training"]["max_audio_duration"]

    for epoch in range(starting_epoch, num_train_epochs):
        model.train()
        total_loss, total_val_loss = 0, 0
        for step, batch in enumerate(train_dataloader):

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                device = model.device
                text, audios, duration, _ = batch

                with torch.no_grad():
                    audio_list = []

                    for audio_path in audios:

                        wav = read_wav_file(
                            audio_path, length
                        )  ## Only read the first 30 seconds of audio
                        if (
                            wav.shape[0] == 1
                        ):  ## If this audio is mono, we repeat the channel so it become "fake stereo"
                            wav = wav.repeat(2, 1)
                        audio_list.append(wav)

                    audio_input = torch.stack(audio_list, dim=0)
                    audio_input = audio_input.to(device)
                    unwrapped_vae = accelerator.unwrap_model(vae)

                    duration = torch.tensor(duration, device=device)
                    duration = torch.clamp(
                        duration, max=length
                    )  ## clamp duration to max audio length

                    audio_latent = unwrapped_vae.encode(
                        audio_input
                    ).latent_dist.sample()
                    audio_latent = audio_latent.transpose(
                        1, 2
                    )  ## Tranpose  to (bsz, seq_len, channel)

                loss, _, _, _ = model(audio_latent, text, duration=duration)
                total_loss += loss.detach().float()
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                optimizer.step()
                lr_scheduler.step()

            if completed_steps % 10 == 0 and accelerator.is_main_process:

                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2

                total_norm = total_norm**0.5
                logger.info(
                    f"Step {completed_steps}, Loss: {loss.item()}, Grad Norm: {total_norm}"
                )

                lr = lr_scheduler.get_last_lr()[0]
                result = {
                    "train_loss": loss.item(),
                    "grad_norm": total_norm,
                    "learning_rate": lr,
                }

                # result["val_loss"] = round(total_val_loss.item()/len(eval_dataloader), 4)
                wandb.log(result, step=completed_steps)

            # Checks if the accelerator has performed an optimization step behind the scenes

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps }"
                    if output_dir is not None:
                        output_dir = os.path.join(output_dir, output_dir)
                    accelerator.save_state(output_dir)

        if completed_steps >= args.max_train_steps:
            break

        model.eval()
        eval_progress_bar = tqdm(
            range(len(eval_dataloader)), disable=not accelerator.is_local_main_process
        )
        for step, batch in enumerate(eval_dataloader):
            with accelerator.accumulate(model) and torch.no_grad():
                device = model.device
                text, audios, duration, _ = batch

                audio_list = []
                for audio_path in audios:

                    wav = read_wav_file(
                        audio_path, length
                    )  ## make sure none of audio exceed 30 sec
                    if (
                        wav.shape[0] == 1
                    ):  ## If this audio is mono, we repeat the channel so it become "fake stereo"
                        wav = wav.repeat(2, 1)
                    audio_list.append(wav)

                audio_input = torch.stack(audio_list, dim=0)
                audio_input = audio_input.to(device)
                duration = torch.tensor(duration, device=device)
                unwrapped_vae = accelerator.unwrap_model(vae)
                audio_latent = unwrapped_vae.encode(audio_input).latent_dist.sample()
                audio_latent = audio_latent.transpose(
                    1, 2
                )  ## Tranpose  to (bsz, seq_len, channel)

                val_loss, _, _, _ = model(audio_latent, text, duration=duration)

                total_val_loss += val_loss.detach().float()
                eval_progress_bar.update(1)

        if accelerator.is_main_process:

            result = {}
            result["epoch"] = float(epoch + 1)

            result["epoch/train_loss"] = round(
                total_loss.item() / len(train_dataloader), 4
            )
            result["epoch/val_loss"] = round(
                total_val_loss.item() / len(eval_dataloader), 4
            )

            wandb.log(result, step=completed_steps)

            result_string = "Epoch: {}, Loss Train: {}, Val: {}\n".format(
                epoch, result["epoch/train_loss"], result["epoch/val_loss"]
            )

            accelerator.print(result_string)

            with open("{}/summary.jsonl".format(output_dir), "a") as f:
                f.write(json.dumps(result) + "\n\n")

            logger.info(result)

            if result["epoch/val_loss"] < best_loss:
                best_loss = result["epoch/val_loss"]
                save_checkpoint = True
            else:
                save_checkpoint = False

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and args.checkpointing_steps == "best":
            if save_checkpoint:
                accelerator.save_state("{}/{}".format(output_dir, "best"))

            if (epoch + 1) % args.save_every == 0:
                accelerator.save_state(
                    "{}/{}".format(output_dir, "epoch_" + str(epoch + 1))
                )

        if accelerator.is_main_process and args.checkpointing_steps == "epoch":
            accelerator.save_state(
                "{}/{}".format(output_dir, "epoch_" + str(epoch + 1))
            )


if __name__ == "__main__":
    main()
