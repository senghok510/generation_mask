from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from storyboard.features import FeatureBank, encode_image_paths, encode_texts
from storyboard.models.consistency_scorer import StoryConsistencyScorer, load_checkpoint
from storyboard.utils.common import ensure_dir, resolve_device, write_json
from storyboard.viz import save_storyboard_grid


def _build_ip_adapter_image_embeds(pipeline, reference_image: Image.Image) -> list[torch.Tensor] | None:
    if getattr(pipeline, "image_encoder", None) is None or getattr(pipeline, "feature_extractor", None) is None:
        return None

    projection_layers = getattr(getattr(pipeline.unet, "encoder_hid_proj", None), "image_projection_layers", [])
    if len(projection_layers) != 1:
        return None

    projection_layer = projection_layers[0]
    if type(projection_layer).__name__ != "IPAdapterPlusImageProjection":
        return None

    dtype = next(pipeline.image_encoder.parameters()).dtype
    pixel_values = pipeline.feature_extractor(reference_image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device=pipeline.device, dtype=dtype)

    image_hidden_states = pipeline.image_encoder(pixel_values, output_hidden_states=True).hidden_states[-2]
    negative_hidden_states = pipeline.image_encoder(
        torch.zeros_like(pixel_values),
        output_hidden_states=True,
    ).hidden_states[-2]

    image_embeds = pipeline.image_encoder.visual_projection(image_hidden_states)
    negative_image_embeds = pipeline.image_encoder.visual_projection(negative_hidden_states)

    # Diffusers expects per-adapter tensors shaped as
    # [batch, num_images, sequence_length, embed_dim] for IP-Adapter Plus.
    combined = torch.cat([negative_image_embeds, image_embeds], dim=0).unsqueeze(1)
    return [combined]


class DiffusersStoryboardGenerator:
    def __init__(
        self,
        *,
        model_id: str,
        device: str | None = None,
        use_ip_adapter: bool = False,
        ip_adapter_repo: str | None = None,
        ip_adapter_subfolder: str = "sdxl_models",
        ip_adapter_weight_name: str = "ip-adapter-plus_sdxl_vit-h.safetensors",
        ip_adapter_image_encoder_subfolder: str | None = None,
        torch_dtype: str = "float16",
    ) -> None:
        from diffusers import AutoPipelineForText2Image
        from transformers import CLIPVisionModelWithProjection

        resolved_device, resolved_dtype = resolve_device(prefer_half=True)
        self.device = device or resolved_device
        dtype = resolved_dtype if self.device == resolved_device else (torch.float16 if torch_dtype == "float16" else torch.float32)
        if self.device == "cpu":
            dtype = torch.float32
        self.use_ip_adapter = use_ip_adapter
        pipeline_kwargs = {"torch_dtype": dtype}
        if self.use_ip_adapter:
            if not ip_adapter_repo:
                raise ValueError("--ip-adapter-repo is required when --use-ip-adapter is set")
            image_encoder_subfolder = ip_adapter_image_encoder_subfolder
            if image_encoder_subfolder is None and "plus" in ip_adapter_weight_name:
                image_encoder_subfolder = f"{ip_adapter_subfolder}/image_encoder"
            if image_encoder_subfolder:
                pipeline_kwargs["image_encoder"] = CLIPVisionModelWithProjection.from_pretrained(
                    ip_adapter_repo,
                    subfolder=image_encoder_subfolder,
                    torch_dtype=dtype,
                )
        self.pipeline = AutoPipelineForText2Image.from_pretrained(model_id, **pipeline_kwargs)
        self.pipeline = self.pipeline.to(self.device)
        if self.use_ip_adapter:
            self.pipeline.load_ip_adapter(
                ip_adapter_repo,
                subfolder=ip_adapter_subfolder,
                weight_name=ip_adapter_weight_name,
            )

    def generate_candidates(
        self,
        *,
        story_id: str,
        reference_image_path: str,
        captions: list[str],
        output_dir: str,
        num_candidates: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
        negative_prompt: str | None = None,
        ip_adapter_scale: float = 0.8,
    ) -> dict:
        output_path = ensure_dir(Path(output_dir) / story_id)
        reference_image = Image.open(reference_image_path).convert("RGB")
        if self.use_ip_adapter:
            self.pipeline.set_ip_adapter_scale(ip_adapter_scale)
        bank = {
            "story_id": story_id,
            "reference_image_path": reference_image_path,
            "panels": [],
        }
        for panel_index, caption in enumerate(captions):
            panel_candidates = []
            for candidate_index in range(num_candidates):
                generator = torch.Generator(device=self.device).manual_seed(seed + panel_index * 1000 + candidate_index)
                kwargs = {
                    "prompt": caption,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": num_inference_steps,
                    "guidance_scale": guidance_scale,
                    "generator": generator,
                }
                if self.use_ip_adapter:
                    ip_adapter_image_embeds = _build_ip_adapter_image_embeds(self.pipeline, reference_image)
                    if ip_adapter_image_embeds is None:
                        kwargs["ip_adapter_image"] = reference_image
                    else:
                        kwargs["ip_adapter_image_embeds"] = ip_adapter_image_embeds
                result = self.pipeline(**kwargs)
                image = result.images[0]
                image_path = output_path / f"panel_{panel_index:02d}_candidate_{candidate_index:02d}.png"
                image.save(image_path)
                panel_candidates.append(str(image_path))
            bank["panels"].append({"caption": caption, "candidate_paths": panel_candidates})
        reference_image.close()
        return bank


def _score_sequence(
    scorer: StoryConsistencyScorer,
    reference_embedding: torch.Tensor,
    image_embeddings: list[torch.Tensor],
    text_embeddings: list[torch.Tensor],
) -> float:
    with torch.inference_mode():
        output = scorer(
            reference_embedding.unsqueeze(0),
            torch.stack(image_embeddings).unsqueeze(0),
            torch.stack(text_embeddings).unsqueeze(0),
            torch.ones(1, len(image_embeddings)),
        )
    return float(output["story_score"].item())


def rerank_candidate_bank(
    candidate_bank: list[dict],
    *,
    scorer_checkpoint: str,
    clip_model_name: str,
    beam_size: int = 4,
    output_dir: str,
) -> list[dict]:
    ensure_dir(output_dir)
    scorer, _ = load_checkpoint(scorer_checkpoint)
    scorer.eval()
    image_paths = []
    captions = []
    for story in candidate_bank:
        image_paths.append(story["reference_image_path"])
        for panel in story["panels"]:
            captions.append(panel["caption"])
            image_paths.extend(panel["candidate_paths"])
    feature_bank = FeatureBank(
        clip_model_name=clip_model_name,
        image_features=encode_image_paths(image_paths, clip_model_name=clip_model_name),
        text_features=encode_texts(captions, clip_model_name=clip_model_name),
    )

    selected_stories: list[dict] = []
    for story in candidate_bank:
        reference_embedding = feature_bank.get_image(story["reference_image_path"]).float()
        text_embeddings = [feature_bank.get_text(panel["caption"]).float() for panel in story["panels"]]
        beams = [([], 0.0)]
        for panel_index, panel in enumerate(story["panels"]):
            next_beams = []
            for prefix, _ in beams:
                for candidate_path in panel["candidate_paths"]:
                    candidate_embeddings = prefix + [feature_bank.get_image(candidate_path).float()]
                    score = _score_sequence(
                        scorer,
                        reference_embedding,
                        candidate_embeddings,
                        text_embeddings[: len(candidate_embeddings)],
                    )
                    next_beams.append((candidate_embeddings, score, prefix + [candidate_path]))
            next_beams = sorted(next_beams, key=lambda item: item[1], reverse=True)[:beam_size]
            beams = [(item[2], item[1]) for item in next_beams]
        best_paths, best_score = max(beams, key=lambda item: item[1])
        selected_story = {
            "story_id": story["story_id"],
            "reference_image_path": story["reference_image_path"],
            "selection_score": best_score,
            "panels": [
                {"caption": panel["caption"], "image_path": image_path}
                for panel, image_path in zip(story["panels"], best_paths)
            ],
        }
        selected_stories.append(selected_story)
        save_storyboard_grid(
            reference_image_path=selected_story["reference_image_path"],
            panels=selected_story["panels"],
            output_path=Path(output_dir) / f"{selected_story['story_id']}.png",
            title=f"{selected_story['story_id']} | score={best_score:.3f}",
        )
    write_json({"stories": selected_stories}, Path(output_dir) / "selected_storyboards.json")
    return selected_stories
