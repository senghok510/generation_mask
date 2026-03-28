from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from facemask_utils import pair_images_by_stem
except ModuleNotFoundError:
    from scripts.facemask_utils import pair_images_by_stem

try:
    from peft import LoraConfig
except ModuleNotFoundError:
    LoraConfig = None


LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


ATTENTION_TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
]


BROAD_TARGET_MODULES = [
    "conv_in",
    "conv_out",
    "proj_in",
    "proj_out",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "ff.net.0.proj",
    "ff.net.2",
    "conv1",
    "conv2",
    "conv_shortcut",
    "downsamplers.0.conv",
    "upsamplers.0.conv",
    "time_emb_proj",
    "linear_1",
    "linear_2",
]


@dataclass(frozen=True)
class TrainConfig:
    pretrained_model_name_or_path: str
    data_root: str
    split: str
    output_dir: str
    prompt: str
    resolution: int
    train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lr_scheduler: str
    lr_warmup_steps: int
    max_grad_norm: float
    num_train_epochs: int
    max_train_steps: int
    checkpointing_steps: int
    checkpointing_epochs: int | None
    mixed_precision: str
    seed: int
    lora_rank: int
    lora_alpha: int
    lora_target_mode: str
    num_workers: int
    gradient_checkpointing: bool
    allow_tf32: bool
    local_files_only: bool
    resume_from_checkpoint: str | None


@dataclass(frozen=True)
class ResumeState:
    checkpoint_dir: Path
    global_step: int
    epoch: int
    micro_batches_seen_in_epoch: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a pretrained Stable Diffusion UNet as a full-image conditional "
            "diffusion model. The conditioning image is the clean face and the target "
            "image is the masked face."
        )
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="runwayml/stable-diffusion-inpainting",
        help="Diffusers model id or local path for the pretrained backbone.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets") / "facemask_aligned" / "paired",
        help="Root containing train/clean and train/masked directories.",
    )
    parser.add_argument("--split", default="train", help="Dataset split to use for fine-tuning.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("checkpoints") / "sd_fullframe_lora_face_mask",
        help="Where checkpoints, metadata, and final LoRA weights are stored.",
    )
    parser.add_argument(
        "--prompt",
        default="a high quality portrait photo of a person wearing a protective face mask",
        help="Text prompt used during training and later generation.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="Training image size. 256 is the safer default before attempting 512.",
    )
    parser.add_argument("--train-batch-size", type=int, default=1, help="Per-device batch size.")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=4,
        help="Accumulate gradients to reach a larger effective batch size.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="AdamW learning rate.")
    parser.add_argument(
        "--lr-scheduler",
        default="constant",
        choices=("constant", "cosine", "cosine_with_restarts", "linear", "polynomial"),
        help="Learning-rate schedule.",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=0, help="Warmup steps for the LR scheduler.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping value.")
    parser.add_argument("--num-train-epochs", type=int, default=20, help="Total number of training epochs.")
    parser.add_argument(
        "--max-train-steps",
        type=int,
        help="Optional override for total update steps. Defaults to epochs * steps_per_epoch.",
    )
    parser.add_argument(
        "--checkpointing-steps",
        type=int,
        default=1000,
        help="How often to save resumable numbered checkpoints.",
    )
    parser.add_argument(
        "--checkpointing-epochs",
        type=int,
        help="Optional epoch interval for checkpointing. When set, this overrides --checkpointing-steps.",
    )
    parser.add_argument(
        "--mixed-precision",
        default="fp16",
        choices=("no", "fp16", "bf16"),
        help="Accelerate mixed precision mode.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank.")
    parser.add_argument("--lora-alpha", type=int, default=8, help="LoRA alpha.")
    parser.add_argument(
        "--lora-target-mode",
        default="broad",
        choices=("attention", "broad"),
        help="Whether LoRA touches only attention projections or a much broader slice of the UNet.",
    )
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        help="Enable gradient checkpointing on the UNet to reduce memory.",
    )
    parser.add_argument(
        "--allow-tf32",
        action="store_true",
        help="Enable TF32 matmuls on Ampere+ GPUs for a small speed boost.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load all pretrained components only from the local Hugging Face cache.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        help="Optional checkpoint directory containing lora/, training_state.pt, and train_state.json.",
    )
    return parser.parse_args()


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1.0


class PairedFullFrameDataset(Dataset):
    def __init__(self, split_root: Path, resolution: int) -> None:
        self.clean_dir = split_root / "clean"
        self.masked_dir = split_root / "masked"
        self.examples = pair_images_by_stem(self.clean_dir, self.masked_dir)
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        stem, clean_path, masked_path = self.examples[index]
        clean = Image.open(clean_path).convert("RGB").resize(
            (self.resolution, self.resolution), resample=Image.Resampling.LANCZOS
        )
        masked = Image.open(masked_path).convert("RGB").resize(
            (self.resolution, self.resolution), resample=Image.Resampling.LANCZOS
        )

        return {
            "stem": stem,
            "clean_image": image_to_tensor(clean),
            "target_image": image_to_tensor(masked),
        }


def collate_fn(examples: list[dict[str, torch.Tensor | str]]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "stem": [example["stem"] for example in examples],
        "clean_image": torch.stack([example["clean_image"] for example in examples]).contiguous().float(),
        "target_image": torch.stack([example["target_image"] for example in examples]).contiguous().float(),
    }


def get_target_modules(target_mode: str) -> list[str]:
    if target_mode == "attention":
        return ATTENTION_TARGET_MODULES
    if target_mode == "broad":
        return BROAD_TARGET_MODULES
    raise ValueError(f"Unsupported LoRA target mode: {target_mode}")


def add_lora_adapter(unet: UNet2DConditionModel, rank: int, alpha: int, target_mode: str) -> list[torch.nn.Parameter]:
    if LoraConfig is None:
        raise ImportError(
            "PEFT is not installed in the active interpreter. Install it with "
            "`python -m pip install peft` and rerun the training command."
        )

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=get_target_modules(target_mode),
    )
    unet.add_adapter(lora_config)
    return [parameter for parameter in unet.parameters() if parameter.requires_grad]


def build_dataloader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        shuffle=True,
        generator=generator,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=collate_fn,
    )


def write_json(path: Path, payload: dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def save_lora_export(unet: UNet2DConditionModel, output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_name = "default"
    peft_config = getattr(unet, "peft_config", None)
    if peft_config:
        adapter_name = next(iter(peft_config))
    unet.save_lora_adapter(output_dir, adapter_name=adapter_name, weight_name=LORA_WEIGHT_NAME)


def save_resumable_checkpoint(
    accelerator: Accelerator,
    unet: UNet2DConditionModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_dir: Path,
    config: TrainConfig,
    global_step: int,
    epoch: int,
    micro_batches_seen_in_epoch: int,
) -> None:
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_unet = accelerator.unwrap_model(unet)
    save_lora_export(unwrapped_unet, checkpoint_dir / "lora")

    metadata = {
        **asdict(config),
        "global_step": global_step,
        "epoch": epoch,
        "micro_batches_seen_in_epoch": micro_batches_seen_in_epoch,
    }
    write_json(checkpoint_dir / "train_state.json", metadata)

    training_state = {
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "global_step": global_step,
        "epoch": epoch,
        "micro_batches_seen_in_epoch": micro_batches_seen_in_epoch,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "python_rng_state": random.getstate(),
    }
    if torch.cuda.is_available():
        training_state["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    if getattr(accelerator, "scaler", None) is not None:
        training_state["grad_scaler"] = accelerator.scaler.state_dict()

    accelerator.save(training_state, checkpoint_dir / "training_state.pt")
    (config_output_dir := Path(config.output_dir)).mkdir(parents=True, exist_ok=True)
    (config_output_dir / "latest_checkpoint.txt").write_text(str(checkpoint_dir.resolve()), encoding="utf-8")


def resolve_resume_checkpoint(resume_from_checkpoint: Path | None) -> ResumeState | None:
    if resume_from_checkpoint is None:
        return None

    checkpoint_dir = resume_from_checkpoint.resolve()
    train_state_path = checkpoint_dir / "train_state.json"
    if not train_state_path.exists():
        raise FileNotFoundError(f"Missing train_state.json in {checkpoint_dir}")

    with open(train_state_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    return ResumeState(
        checkpoint_dir=checkpoint_dir,
        global_step=int(metadata.get("global_step", 0)),
        epoch=int(metadata.get("epoch", 0)),
        micro_batches_seen_in_epoch=int(metadata.get("micro_batches_seen_in_epoch", 0)),
    )


def load_lora_into_unet(unet: UNet2DConditionModel, checkpoint_dir: Path) -> None:
    direct_weight = checkpoint_dir / LORA_WEIGHT_NAME
    nested_weight = checkpoint_dir / "lora" / LORA_WEIGHT_NAME

    if direct_weight.exists():
        unet.load_lora_adapter(checkpoint_dir, prefix=None, weight_name=LORA_WEIGHT_NAME)
        return
    if nested_weight.exists():
        unet.load_lora_adapter(checkpoint_dir / "lora", prefix=None, weight_name=LORA_WEIGHT_NAME)
        return
    raise FileNotFoundError(
        f"Could not find {LORA_WEIGHT_NAME} in {checkpoint_dir} or {checkpoint_dir / 'lora'}"
    )


def load_optimizer_state(
    accelerator: Accelerator,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    checkpoint_dir: Path,
) -> None:
    training_state_path = checkpoint_dir / "training_state.pt"
    if not training_state_path.exists():
        raise FileNotFoundError(f"Missing training_state.pt in {checkpoint_dir}")

    state = torch.load(training_state_path, map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    lr_scheduler.load_state_dict(state["lr_scheduler"])

    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    if "python_rng_state" in state:
        random.setstate(state["python_rng_state"])
    if torch.cuda.is_available() and "cuda_rng_state_all" in state:
        torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
    if getattr(accelerator, "scaler", None) is not None and "grad_scaler" in state:
        accelerator.scaler.load_state_dict(state["grad_scaler"])


def main() -> None:
    args = parse_args()
    split_root = args.data_root / args.split
    if not split_root.exists():
        raise FileNotFoundError(f"Split root does not exist: {split_root}")
    if args.checkpointing_steps <= 0:
        raise ValueError("--checkpointing-steps must be a positive integer.")
    if args.checkpointing_epochs is not None and args.checkpointing_epochs <= 0:
        raise ValueError("--checkpointing-epochs must be a positive integer.")

    if args.allow_tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    pretrained_kwargs = {"local_files_only": args.local_files_only}
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer", **pretrained_kwargs
    )
    text_encoder = CLIPTextModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", **pretrained_kwargs
    )
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae", **pretrained_kwargs)
    unet = UNet2DConditionModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="unet", **pretrained_kwargs
    )
    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", **pretrained_kwargs
    )

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    if args.gradient_checkpointing:
        unet.enable_gradient_checkpointing()

    resume_state = resolve_resume_checkpoint(args.resume_from_checkpoint)
    if resume_state is not None:
        load_lora_into_unet(unet, resume_state.checkpoint_dir)
    else:
        add_lora_adapter(
            unet,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_mode=args.lora_target_mode,
        )

    lora_parameters = [parameter for parameter in unet.parameters() if parameter.requires_grad]

    optimizer = torch.optim.AdamW(
        lora_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    dataset = PairedFullFrameDataset(split_root=split_root, resolution=args.resolution)
    micro_batches_per_epoch = math.ceil(len(dataset) / args.train_batch_size)
    num_update_steps_per_epoch = math.ceil(micro_batches_per_epoch / args.gradient_accumulation_steps)
    max_train_steps = args.max_train_steps or args.num_train_epochs * num_update_steps_per_epoch
    checkpointing_interval = (
        args.checkpointing_epochs * num_update_steps_per_epoch
        if args.checkpointing_epochs is not None
        else args.checkpointing_steps
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=max_train_steps * args.gradient_accumulation_steps,
    )

    unet, optimizer, lr_scheduler = accelerator.prepare(unet, optimizer, lr_scheduler)

    if resume_state is not None:
        load_optimizer_state(accelerator, optimizer, lr_scheduler, resume_state.checkpoint_dir)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    prompt_ids = tokenizer(
        args.prompt,
        truncation=True,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    with torch.no_grad():
        prompt_embeds = text_encoder(prompt_ids)[0]

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    config = TrainConfig(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        data_root=str(args.data_root.resolve()),
        split=args.split,
        output_dir=str(args.output_dir.resolve()),
        prompt=args.prompt,
        resolution=args.resolution,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_steps=args.lr_warmup_steps,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        max_train_steps=max_train_steps,
        checkpointing_steps=checkpointing_interval,
        checkpointing_epochs=args.checkpointing_epochs,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_target_mode=args.lora_target_mode,
        num_workers=args.num_workers,
        gradient_checkpointing=args.gradient_checkpointing,
        allow_tf32=args.allow_tf32,
        local_files_only=args.local_files_only,
        resume_from_checkpoint=str(args.resume_from_checkpoint.resolve()) if args.resume_from_checkpoint else None,
    )

    if accelerator.is_main_process:
        write_json(args.output_dir / "train_config.json", asdict(config))
        lora_param_count = sum(parameter.numel() for parameter in accelerator.unwrap_model(unet).parameters() if parameter.requires_grad)
        print(json.dumps(asdict(config), indent=2))
        print(f"Dataset size: {len(dataset)}")
        print(f"Micro-batches per epoch: {micro_batches_per_epoch}")
        print(f"Update steps per epoch: {num_update_steps_per_epoch}")
        print(f"Total batch size: {total_batch_size}")
        print(f"Trainable LoRA parameters: {lora_param_count}")

    global_step = resume_state.global_step if resume_state is not None else 0
    start_epoch = resume_state.epoch if resume_state is not None else 0
    resume_micro_batches = resume_state.micro_batches_seen_in_epoch if resume_state is not None else 0

    if resume_micro_batches >= micro_batches_per_epoch:
        start_epoch += 1
        resume_micro_batches = 0

    progress_bar = tqdm(
        total=max_train_steps,
        initial=global_step,
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Training")

    if global_step >= max_train_steps:
        if accelerator.is_main_process:
            print("The requested training budget was already reached by the resume checkpoint.")
        accelerator.end_training()
        return

    train_loss = 0.0
    final_epoch = start_epoch

    for epoch in range(start_epoch, args.num_train_epochs):
        final_epoch = epoch
        epoch_seed = args.seed + epoch if args.seed is not None else epoch
        train_dataloader = build_dataloader(
            dataset=dataset,
            batch_size=args.train_batch_size,
            num_workers=args.num_workers,
            seed=epoch_seed,
        )

        unet.train()
        for step, batch in enumerate(train_dataloader):
            if epoch == start_epoch and step < resume_micro_batches:
                continue

            micro_batches_seen_in_epoch = step + 1
            with accelerator.accumulate(unet):
                target_images = batch["target_image"].to(device=accelerator.device, dtype=weight_dtype)
                clean_images = batch["clean_image"].to(device=accelerator.device, dtype=weight_dtype)

                with torch.no_grad():
                    target_latents = vae.encode(target_images).latent_dist.sample() * vae.config.scaling_factor
                    clean_latents = vae.encode(clean_images).latent_dist.sample() * vae.config.scaling_factor
                    latent_masks = torch.zeros(
                        (target_latents.shape[0], 1, target_latents.shape[2], target_latents.shape[3]),
                        device=target_latents.device,
                        dtype=target_latents.dtype,
                    )

                noise = torch.randn_like(target_latents)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (target_latents.shape[0],),
                    device=target_latents.device,
                    dtype=torch.long,
                )
                noisy_latents = noise_scheduler.add_noise(target_latents, noise, timesteps)

                model_input = torch.cat([noisy_latents, latent_masks, clean_latents], dim=1)
                encoder_hidden_states = prompt_embeds.repeat(target_latents.shape[0], 1, 1)
                model_pred = unet(model_input, timesteps, encoder_hidden_states=encoder_hidden_states, return_dict=False)[0]

                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(target_latents, noise, timesteps)
                else:
                    raise ValueError(f"Unsupported prediction type: {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                avg_loss = accelerator.gather(loss.detach().repeat(target_latents.shape[0])).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(lora_parameters, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                progress_bar.update(1)
                progress_bar.set_postfix(loss=f"{train_loss:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                if global_step % checkpointing_interval == 0:
                    checkpoint_dir = args.output_dir / f"checkpoint-{global_step}"
                    save_resumable_checkpoint(
                        accelerator=accelerator,
                        unet=unet,
                        optimizer=optimizer,
                        lr_scheduler=lr_scheduler,
                        checkpoint_dir=checkpoint_dir,
                        config=config,
                        global_step=global_step,
                        epoch=epoch,
                        micro_batches_seen_in_epoch=micro_batches_seen_in_epoch,
                    )

                train_loss = 0.0

            if global_step >= max_train_steps:
                break

        resume_micro_batches = 0
        if global_step >= max_train_steps:
            break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        final_dir = args.output_dir / "final"
        unwrapped_unet = accelerator.unwrap_model(unet)
        save_lora_export(unwrapped_unet, final_dir)
        write_json(
            final_dir / "train_state.json",
            {
                **asdict(config),
                "global_step": global_step,
                "epoch": final_epoch,
                "micro_batches_seen_in_epoch": 0,
            },
        )
        print(f"Saved final LoRA weights to {final_dir.resolve()}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
