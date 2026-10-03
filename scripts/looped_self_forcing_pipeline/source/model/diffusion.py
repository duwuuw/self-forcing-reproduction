from typing import Tuple

import torch

from model.base import BaseModel
from model.lora import LoraTrainingConfig
from pipeline import SelfForcingTrainingPipeline
from utils.temporal_loop import TemporalLoopConfig
from utils.training_config import validate_supervised_training_preflight
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class CausalDiffusion(BaseModel):
    def __init__(self, args, device):
        validate_supervised_training_preflight(args)
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
        self.independent_first_frame = getattr(args, "independent_first_frame", False)

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.teacher_forcing = True
        self.inference_pipeline = None

    def _initialize_models(self, args, device):
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True
        )
        self.temporal_loop = TemporalLoopConfig.from_config(
            args, runtime_num_layers=len(self.generator.model.blocks)
        )
        self.lora_config = LoraTrainingConfig.from_config(args)
        self.lora_enabled = self.lora_config.enabled
        self.lora_parameter_names = ()
        self.base_checkpoint_loaded_before_lora = False
        self.generator.model.requires_grad_(not self.lora_enabled)

        self.text_encoder = WanTextEncoder().requires_grad_(False)
        self.vae = WanVAEWrapper().requires_grad_(False)
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_training_pipeline(self):
        if self.inference_pipeline is None:
            self.inference_pipeline = SelfForcingTrainingPipeline(
                denoising_step_list=self.denoising_step_list.tolist(),
                scheduler=self.scheduler,
                generator=self.generator,
                num_frame_per_block=self.num_frame_per_block,
                independent_first_frame=self.independent_first_frame,
                same_step_across_blocks=getattr(
                    self.args, "same_step_across_blocks", False
                ),
                last_step_only=getattr(self.args, "last_step_only", False),
                num_training_frames=self.num_training_frames,
                context_noise=0,
                min_training_frames=self.min_training_frames,
                training_gradient_window_frames=(
                    self.training_gradient_window_frames
                ),
                total_ar_blocks=self.total_ar_blocks,
                temporal_loop=self.temporal_loop,
                i2v=getattr(self.args, "i2v", False),
            )
        return self.inference_pipeline

    def generator_loss(
        self,
        clean_latent: torch.Tensor,
        conditional_dict: dict,
    ) -> Tuple[torch.Tensor, dict]:
        noise = torch.randn_like(clean_latent)
        batch_size, num_frames = clean_latent.shape[:2]
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            batch_size,
            num_frames,
            self.num_frame_per_block,
            uniform_timestep=False,
        )
        timestep = self.scheduler.timesteps[index].to(
            dtype=self.dtype, device=self.device
        )
        noisy_latent = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frames))
        training_target = self.scheduler.training_target(
            clean_latent, noise, timestep
        )

        flow_pred = self._get_training_pipeline().teacher_forced_flow_prediction(
            clean_latent=clean_latent,
            noisy_latent=noisy_latent,
            timestep=timestep,
            conditional_dict=conditional_dict,
        )
        per_frame_loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction="none"
        ).mean(dim=(2, 3, 4))
        weights = self.scheduler.training_weight(timestep).unflatten(
            0, (batch_size, num_frames)
        )
        loss = (per_frame_loss * weights).mean()

        log_dict = {
            "x0": clean_latent.detach(),
            "flow_pred": flow_pred.detach(),
        }
        return loss, log_dict
