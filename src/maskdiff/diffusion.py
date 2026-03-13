from __future__ import annotations

import math

import torch


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


class DiffusionSchedule:
    def __init__(self, timesteps: int = 250, device: torch.device | None = None) -> None:
        self.timesteps = timesteps
        self.device = device or torch.device("cpu")

        betas = cosine_beta_schedule(timesteps).to(self.device)
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        alpha_cumprod_prev = torch.cat([torch.ones(1, device=self.device), alpha_cumprod[:-1]], dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_cumprod = alpha_cumprod
        self.alpha_cumprod_prev = alpha_cumprod_prev
        self.sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - alpha_cumprod)
        self.sqrt_recip_alpha_cumprod = torch.sqrt(1.0 / alpha_cumprod)
        self.sqrt_recipm1_alpha_cumprod = torch.sqrt(1.0 / alpha_cumprod - 1)
        self.posterior_variance = betas * (1.0 - alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.posterior_mean_coef1 = betas * torch.sqrt(alpha_cumprod_prev) / (1.0 - alpha_cumprod)
        self.posterior_mean_coef2 = (1.0 - alpha_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alpha_cumprod)

    def to(self, device: torch.device) -> "DiffusionSchedule":
        for name, value in vars(self).items():
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        self.device = device
        return self

    def _extract(self, values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        gathered = values[timesteps].to(target.device, target.dtype)
        return gathered.view(-1, 1, 1, 1)

    def q_sample(self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None) -> torch.Tensor:
        noise = torch.randn_like(clean) if noise is None else noise
        return self._extract(self.sqrt_alpha_cumprod, timesteps, clean) * clean + self._extract(
            self.sqrt_one_minus_alpha_cumprod, timesteps, clean
        ) * noise

    def predict_start_from_noise(self, noisy: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._extract(self.sqrt_recip_alpha_cumprod, timesteps, noisy) * noisy - self._extract(
            self.sqrt_recipm1_alpha_cumprod, timesteps, noisy
        ) * noise

    def p_mean_variance(
        self,
        model,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        guidance_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        model_input = torch.cat([noisy, condition], dim=1)
        if guidance_scale == 1.0:
            pred_noise = model(model_input, timesteps)
        else:
            zeros = torch.zeros_like(condition)
            pred_uncond = model(torch.cat([noisy, zeros], dim=1), timesteps)
            pred_cond = model(model_input, timesteps)
            pred_noise = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

        x0 = self.predict_start_from_noise(noisy, timesteps, pred_noise).clamp(-1.0, 1.0)
        mean = self._extract(self.posterior_mean_coef1, timesteps, noisy) * x0 + self._extract(
            self.posterior_mean_coef2, timesteps, noisy
        ) * noisy
        variance = self._extract(self.posterior_variance, timesteps, noisy)
        return mean, variance, x0

    @torch.no_grad()
    def sample(
        self,
        model,
        condition: torch.Tensor,
        guidance_scale: float = 2.0,
        clip_denoised: bool = True,
        sample_steps: int | None = None,
    ) -> torch.Tensor:
        batch, _, height, width = condition.shape
        current = torch.randn(batch, 3, height, width, device=condition.device)
        total_steps = self.timesteps if sample_steps is None else max(1, min(sample_steps, self.timesteps))
        schedule = torch.linspace(self.timesteps - 1, 0, steps=total_steps, device=condition.device).long()
        schedule = torch.unique_consecutive(schedule)

        for index, step in enumerate(schedule.tolist()):
            timesteps = torch.full((batch,), step, device=condition.device, dtype=torch.long)
            mean, variance, x0 = self.p_mean_variance(model, current, timesteps, condition, guidance_scale)
            if clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)
            if index < len(schedule) - 1 and step > 0:
                current = mean + torch.sqrt(variance.clamp(min=1e-20)) * torch.randn_like(current)
            else:
                current = x0
        return current
