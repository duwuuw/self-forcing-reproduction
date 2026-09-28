from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist

from utils.temporal_loop import TemporalLoopConfig


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: Optional[int] = None,
                 context_noise: int = 0,
                 min_training_frames: Optional[int] = None,
                 num_training_frames: Optional[int] = None,
                 training_gradient_window_frames: Optional[int] = None,
                 total_ar_blocks: Optional[int] = None,
                 temporal_loop=None,
                 i2v: bool = False,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Training-scale settings retain the baseline values when callers do
        # not provide the new explicit names.
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = i2v
        if num_training_frames is None:
            num_training_frames = num_max_frames
        if num_training_frames is None:
            raise ValueError("num_training_frames must come from merged config")
        self.min_training_frames = (
            num_training_frames if min_training_frames is None else min_training_frames
        )
        self.num_training_frames = num_training_frames
        self.training_gradient_window_frames = (
            self.min_training_frames
            if training_gradient_window_frames is None
            else training_gradient_window_frames
        )
        self.total_ar_blocks = total_ar_blocks
        self._validate_training_frame_settings()

        runtime_num_layers = len(self.generator.model.blocks)
        if isinstance(temporal_loop, TemporalLoopConfig):
            self.temporal_loop = temporal_loop
        else:
            config = temporal_loop
            if config is not None and not hasattr(config, "temporal_loop"):
                config = {"temporal_loop": config}
            self.temporal_loop = TemporalLoopConfig.from_config(
                config, runtime_num_layers=runtime_num_layers
            )
        if self.temporal_loop.enabled and self.i2v:
            raise ValueError("I2V is not supported when temporal loop is enabled")

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None
        self._kv_cache_signature = None
        self._crossattn_cache_signature = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.frame_seq_length = None

    def _validate_training_frame_settings(self):
        for name in (
            "min_training_frames",
            "num_training_frames",
            "training_gradient_window_frames",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.min_training_frames > self.num_training_frames:
            raise ValueError("min_training_frames cannot exceed num_training_frames")
        if self.training_gradient_window_frames > self.num_training_frames:
            raise ValueError(
                "training_gradient_window_frames cannot exceed num_training_frames"
            )
        if self.total_ar_blocks is not None and (
            isinstance(self.total_ar_blocks, bool)
            or not isinstance(self.total_ar_blocks, int)
            or self.total_ar_blocks < 1
        ):
            raise ValueError("total_ar_blocks must be a positive integer")

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        if dist.is_available() and dist.is_initialized():
            dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if self.temporal_loop.enabled and initial_latent is not None:
            raise ValueError(
                "initial_latent is not supported when temporal loop is enabled"
            )
        if self.temporal_loop.enabled and self.i2v:
            raise ValueError("I2V is not supported when temporal loop is enabled")
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        patch_size = self.generator.model.patch_size
        if height % patch_size[-2] or width % patch_size[-1]:
            raise ValueError("noise spatial dimensions must be divisible by model.patch_size")
        self.frame_seq_length = (
            (height // patch_size[-2]) * (width // patch_size[-1])
        )
        self._cache_noise = noise
        self._cache_num_frames = num_output_frames
        self._cache_context = conditional_dict["prompt_embeds"]

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length
                )
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        actual_total_ar_blocks = len(all_num_frames)
        if (
            self.total_ar_blocks is not None
            and self.total_ar_blocks != actual_total_ar_blocks
        ):
            raise ValueError(
                "configured total_ar_blocks must match actual all_num_frames "
                f"length ({actual_total_ar_blocks})"
            )
        total_ar_blocks = self.total_ar_blocks or actual_total_ar_blocks
        start_gradient_frame_index = (
            num_output_frames - self.training_gradient_window_frames
        )

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            temporal_loop_plan = (
                self.temporal_loop.plan_for_block(block_index, total_ar_blocks)
                if self.temporal_loop.enabled else None
            )
            temporal_loop_kwargs = (
                {"temporal_loop_plan": temporal_loop_plan}
                if temporal_loop_plan is not None else {}
            )
            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = (index == exit_flags[0])
                else:
                    exit_flag = (index == exit_flags[block_index])  # Only backprop at the randomly selected timestep (consistent across all ranks)
                timestep = torch.ones(
                    [batch_size, current_num_frames],
                    device=noise.device,
                    dtype=torch.int64) * current_timestep

                if not exit_flag:
                    with torch.no_grad():
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            **temporal_loop_kwargs
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        noisy_input = self.scheduler.add_noise(
                            denoised_pred.flatten(0, 1),
                            torch.randn_like(denoised_pred.flatten(0, 1)),
                            next_timestep * torch.ones(
                                [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                        ).unflatten(0, denoised_pred.shape[:2])
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    chunk_end_frame = current_start_frame + current_num_frames
                    if chunk_end_frame <= start_gradient_frame_index:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=conditional_dict,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                current_start=current_start_frame * self.frame_seq_length,
                                **temporal_loop_kwargs
                            )
                    else:
                        _, denoised_pred = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep,
                            kv_cache=self.kv_cache1,
                            crossattn_cache=self.crossattn_cache,
                            current_start=current_start_frame * self.frame_seq_length,
                            **temporal_loop_kwargs
                        )
                    break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=conditional_dict,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    **temporal_loop_kwargs
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = self._map_scheduler_timestep(
                self.denoising_step_list[exit_flags[0]], noise.device
            )
        else:
            denoised_timestep_to = self._map_scheduler_timestep(
                self.denoising_step_list[exit_flags[0] + 1], noise.device
            )
            denoised_timestep_from = self._map_scheduler_timestep(
                self.denoising_step_list[exit_flags[0]], noise.device
            )

        if return_sim_step:
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize or reset the causal model's runtime-sized KV cache.
        """
        signature = (batch_size, self._cache_num_frames, self.frame_seq_length, dtype, device)
        if self.kv_cache1 is None or self._kv_cache_signature != signature:
            self.kv_cache1 = self.generator.model.create_kv_cache(
                batch_size=batch_size,
                x=self._cache_noise,
                frame_seqlen=self.frame_seq_length,
                num_frames=self._cache_num_frames,
                dtype=dtype,
                device=device,
            )
            self._kv_cache_signature = signature
        else:
            self._reset_kv_cache(self.kv_cache1)

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize or reset the causal model's runtime-sized cross-attention cache.
        """
        signature = (
            batch_size,
            self._cache_context.shape[1],
            dtype,
            device,
        )
        if self.crossattn_cache is None or self._crossattn_cache_signature != signature:
            self.crossattn_cache = self.generator.model.create_crossattn_cache(
                batch_size=batch_size,
                context=self._cache_context,
                dtype=dtype,
                device=device,
            )
            self._crossattn_cache_signature = signature
        else:
            self._reset_crossattn_cache(self.crossattn_cache)

    def _map_scheduler_timestep(self, timestep, device):
        scheduler_timesteps = self.scheduler.timesteps.to(device)
        timestep = torch.as_tensor(
            timestep, dtype=scheduler_timesteps.dtype, device=device
        )
        return 1000 - torch.argmin(
            (scheduler_timesteps - timestep).abs(), dim=0
        ).item()

    @staticmethod
    def _reset_kv_cache(caches):
        for cache in caches:
            for name in (
                "global_end_index",
                "local_end_index",
                "absolute_start_index",
                "absolute_end_index",
                "physical_start_index",
                "physical_end_index",
            ):
                index = cache.get(name)
                if torch.is_tensor(index):
                    index.zero_()

    @staticmethod
    def _reset_crossattn_cache(caches):
        for cache in caches:
            cache["is_init"] = False
