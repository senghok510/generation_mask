from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKL, DDPMScheduler, DPMSolverMultistepScheduler, UNet2DConditionModel
from huggingface_hub import snapshot_download
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

try:
    from .facemask_utils import pair_images_by_stem
except ImportError:
    from facemask_utils import pair_images_by_stem


LORA_WEIGHT_NAME = "pytorch_lora_weights.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full-image masked-face predictions with the full-frame conditional "
            "Stable Diffusion LoRA model."
        )
    )
    parser.add_argument(
        "--pretrained-model-name-or-path",
        default="runwayml/stable-diffusion-inpainting",
        help="Diffusers model id or local path for the pretrained backbone.",
    )
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("datasets") / "facemask_aligned" / "paired" / "test",
        help="Split root containing clean/ and masked/ directories.",
    )
    parser.add_argument(
        "--lora-dir",
        required=True,
        type=Path,
        help="Path to final/ or checkpoint-*/ containing the full-frame LoRA weights.",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Where generated images are written.")
    parser.add_argument(
        "--prompt",
        help="Prompt used for generation. Defaults to the training prompt when train_config.json is found.",
    )
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Optional negative prompt used for classifier-free guidance.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        help="Inference size. Defaults to the training resolution when available, else 256.",
    )
    parser.add_argument("--num-inference-steps", type=int, default=30, help="Diffusion sampling steps.")
    parser.add_argument("--guidance-scale", type=float, default=7.5, help="Classifier-free guidance scale.")
    parser.add_argument("--seed", type=int, help="Optional seed for reproducible sampling.")
    parser.add_argument("--limit", type=int, help="Optional cap on the number of images to generate.")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load the base model only from the local Hugging Face cache.",
    )
    return parser.parse_args()


def load_train_config(lora_dir: Path) -> dict[str, object]:
    candidates = [
        lora_dir / "train_config.json",
        lora_dir / "train_state.json",
        lora_dir.parent / "train_config.json",
        lora_dir.parent / "train_state.json",
        lora_dir.parent.parent / "train_config.json",
        lora_dir.parent.parent / "train_state.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def resolve_model_root(model_name_or_path: str, local_files_only: bool) -> str:
    candidate = Path(model_name_or_path)
    if candidate.exists():
        return str(candidate.resolve())
    if not local_files_only:
        return model_name_or_path
    return snapshot_download(model_name_or_path, local_files_only=True)


def image_to_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1)[None, ...] / 127.5 - 1.0


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    image = tensor.clamp(-1, 1)
    image = ((image + 1.0) * 127.5).round().byte()
    array = image.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(array, mode="RGB")


def resolve_lora_dir(lora_dir: Path) -> Path:
    direct_weight = lora_dir / LORA_WEIGHT_NAME
    nested_weight = lora_dir / "lora" / LORA_WEIGHT_NAME
    if direct_weight.exists():
        return lora_dir
    if nested_weight.exists():
        return lora_dir / "lora"
    raise FileNotFoundError(
        f"Could not find {LORA_WEIGHT_NAME} in {lora_dir} or {lora_dir / 'lora'}"
    )


def build_prompt_embeddings(
    tokenizer: CLIPTokenizer,
    text_encoder: CLIPTextModel,
    prompt: str,
    negative_prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = tokenizer.model_max_length
    prompt_ids = tokenizer(
        prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).input_ids.to(device)
    negative_ids = tokenizer(
        negative_prompt,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).input_ids.to(device)

    with torch.no_grad():
        prompt_embeds = text_encoder(prompt_ids)[0]
        negative_embeds = text_encoder(negative_ids)[0]
    return prompt_embeds, negative_embeds


def main() -> None:
    args = parse_args()
    clean_dir = args.split_root / "clean"
    masked_dir = args.split_root / "masked"
    if not clean_dir.exists() or not masked_dir.exists():
        raise FileNotFoundError(f"Expected clean/ and masked/ under {args.split_root}")

    train_config = load_train_config(args.lora_dir.resolve())
    prompt = args.prompt or train_config.get(
        "prompt", "a high quality portrait photo of a person wearing a protective face mask"
    )
    resolution = args.resolution or int(train_config.get("resolution", 256))

    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_root = resolve_model_root(args.pretrained_model_name_or_path, args.local_files_only)
    pretrained_kwargs = {"local_files_only": args.local_files_only}

    tokenizer = CLIPTokenizer.from_pretrained(model_root, subfolder="tokenizer", **pretrained_kwargs)
    text_encoder = CLIPTextModel.from_pretrained(
        model_root, subfolder="text_encoder", torch_dtype=torch_dtype, **pretrained_kwargs
    )
    vae = AutoencoderKL.from_pretrained(model_root, subfolder="vae", torch_dtype=torch_dtype, **pretrained_kwargs)
    unet = UNet2DConditionModel.from_pretrained(
        model_root, subfolder="unet", torch_dtype=torch_dtype, **pretrained_kwargs
    )
    train_scheduler = DDPMScheduler.from_pretrained(model_root, subfolder="scheduler", **pretrained_kwargs)
    scheduler = DPMSolverMultistepScheduler.from_config(train_scheduler.config)

    lora_dir = resolve_lora_dir(args.lora_dir.resolve())
    unet.requires_grad_(False)
    unet.load_lora_adapter(lora_dir, prefix=None, weight_name=LORA_WEIGHT_NAME)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    text_encoder.eval()
    vae.eval()
    unet.eval()

    prompt_embeds, negative_embeds = build_prompt_embeddings(
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        prompt=prompt,
        negative_prompt=args.negative_prompt,
        device=device,
    )

    pairs = pair_images_by_stem(clean_dir, masked_dir)
    if args.limit is not None:
        pairs = pairs[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "resolved_model_root": model_root,
        "lora_dir": str(lora_dir.resolve()),
        "split_root": str(args.split_root.resolve()),
        "output_dir": str(args.output_dir.resolve()),
        "prompt": prompt,
        "negative_prompt": args.negative_prompt,
        "resolution": resolution,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "seed": args.seed,
        "count": len(pairs),
        "local_files_only": args.local_files_only,
    }
    with open(args.output_dir / "generation_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    base_seed = args.seed if args.seed is not None else torch.seed()
    for index, (stem, clean_path, _masked_path) in enumerate(tqdm(pairs, desc="Generating")):
        clean = Image.open(clean_path).convert("RGB").resize((resolution, resolution), resample=Image.Resampling.LANCZOS)
        clean_tensor = image_to_tensor(clean).to(device=device, dtype=torch_dtype)

        with torch.no_grad():
            clean_latents = vae.encode(clean_tensor).latent_dist.sample() * vae.config.scaling_factor

        latent_mask = torch.zeros(
            (1, 1, clean_latents.shape[2], clean_latents.shape[3]),
            device=device,
            dtype=clean_latents.dtype,
        )
        scheduler.set_timesteps(args.num_inference_steps, device=device)
        generator = torch.Generator(device=device).manual_seed(int(base_seed) + index)
        latents = torch.randn(clean_latents.shape, generator=generator, device=device, dtype=clean_latents.dtype)
        if hasattr(scheduler, "init_noise_sigma"):
            latents = latents * scheduler.init_noise_sigma

        do_cfg = args.guidance_scale > 1.0
        for timestep in scheduler.timesteps:
            latent_model_input = scheduler.scale_model_input(latents, timestep)

            if do_cfg:
                latent_model_input = torch.cat([latent_model_input, latent_model_input], dim=0)
                model_condition = clean_latents.repeat(2, 1, 1, 1)
                model_mask = latent_mask.repeat(2, 1, 1, 1)
                encoder_hidden_states = torch.cat([negative_embeds, prompt_embeds], dim=0)
            else:
                model_condition = clean_latents
                model_mask = latent_mask
                encoder_hidden_states = prompt_embeds

            model_input = torch.cat([latent_model_input, model_mask, model_condition], dim=1)
            with torch.no_grad():
                noise_pred = unet(
                    model_input,
                    timestep,
                    encoder_hidden_states=encoder_hidden_states,
                    return_dict=False,
                )[0]

            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            latents = scheduler.step(noise_pred, timestep, latents, generator=generator).prev_sample

        with torch.no_grad():
            decoded = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

        result = tensor_to_image(decoded[0].float())
        result.save(args.output_dir / f"{stem}.png")

    print(f"Saved {len(pairs)} generated images to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
