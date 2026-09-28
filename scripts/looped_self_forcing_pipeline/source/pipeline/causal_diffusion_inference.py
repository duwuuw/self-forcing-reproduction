from tqdm import tqdm
from typing import List, Optional
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.temporal_loop import TemporalLoopConfig


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None
    ):
        super().__init__()
        # Step 1: Initialize all models
        self.generator = WanDiffusionWrapper(
            **getattr(args, "model_kwargs", {}), is_causal=True) if generator is None else generator
        self.temporal_loop = TemporalLoopConfig.from_config(
            args, len(self.generator.model.blocks)
        )
        if self.temporal_loop.enabled and (
            getattr(self.generator.model, "model_type", None) == "i2v"
            or getattr(args, "i2v", False)
        ):
            raise ValueError(
                "initial_latent/I2V conditioning is not supported when temporal loop is enabled"
            )
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self._cache_signature = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        if self.temporal_loop.enabled and initial_latent is not None:
            raise ValueError(
                "initial_latent is not supported when temporal loop is enabled"
            )

        batch_size, num_frames, num_channels, height, width = noise.shape
        start_frame_index = 0 if start_frame_index is None else int(start_frame_index)
        if start_frame_index < 0:
            raise ValueError("start_frame_index must be non-negative")
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        patch_size = self.generator.model.patch_size
        if height % patch_size[-2] or width % patch_size[-1]:
            raise ValueError("noise spatial dimensions must be divisible by model.patch_size")
        frame_seq_length = (
            (height // patch_size[-2]) * (width // patch_size[-1])
        )
        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize runtime-sized KV and cross-attention caches.
        cache_origin_frame = start_frame_index
        cache_origin_token = cache_origin_frame * frame_seq_length
        cache_signature = (
            tuple(patch_size),
            (num_channels, height, width),
            frame_seq_length,
            len(self.generator.model.blocks),
            self.local_attn_size,
            batch_size,
            num_output_frames,
            noise.dtype,
            noise.device,
            tuple(conditional_dict["prompt_embeds"].shape),
            tuple(unconditional_dict["prompt_embeds"].shape),
            start_frame_index * frame_seq_length,
            cache_origin_token,
        )
        if self.kv_cache_pos is None or self._cache_signature != cache_signature:
            self.kv_cache_pos = self.generator.model.create_kv_cache(
                batch_size=batch_size,
                x=noise,
                num_frames=num_output_frames,
                dtype=noise.dtype,
                device=noise.device,
                cache_start=cache_origin_token,
            )
            self.kv_cache_neg = self.generator.model.create_kv_cache(
                batch_size=batch_size,
                x=noise,
                num_frames=num_output_frames,
                dtype=noise.dtype,
                device=noise.device,
                cache_start=cache_origin_token,
            )
            self.crossattn_cache_pos = self.generator.model.create_crossattn_cache(
                batch_size=batch_size,
                context=conditional_dict["prompt_embeds"],
                dtype=noise.dtype,
                device=noise.device,
            )
            self.crossattn_cache_neg = self.generator.model.create_crossattn_cache(
                batch_size=batch_size,
                context=unconditional_dict["prompt_embeds"],
                dtype=noise.dtype,
                device=noise.device,
            )
            self._cache_signature = cache_signature
        else:
            self._reset_kv_cache(self.kv_cache_pos)
            self._reset_kv_cache(self.kv_cache_neg)
            self._reset_crossattn_cache(self.crossattn_cache_pos)
            self._reset_crossattn_cache(self.crossattn_cache_neg)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * frame_seq_length,
                    cache_start=cache_origin_token
                )
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * frame_seq_length,
                    cache_start=cache_origin_token
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * frame_seq_length,
                    cache_start=cache_origin_token
                )
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=unconditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * frame_seq_length,
                    cache_start=cache_origin_token
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for block_index, current_num_frames in enumerate(all_num_frames):
            temporal_loop_plan = (
                self.temporal_loop.plan_for_block(block_index, len(all_num_frames))
                if self.temporal_loop.enabled else None
            )
            temporal_loop_kwargs = (
                {"temporal_loop_plan": temporal_loop_plan}
                if temporal_loop_plan is not None else {}
            )
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred_cond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=conditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * frame_seq_length,
                    cache_start=cache_origin_token,
                    **temporal_loop_kwargs
                )
                flow_pred_uncond, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=unconditional_dict,
                    timestep=timestep,
                    kv_cache=self.kv_cache_neg,
                    crossattn_cache=self.crossattn_cache_neg,
                    current_start=current_start_frame * frame_seq_length,
                    cache_start=cache_origin_token,
                    **temporal_loop_kwargs
                )

                flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                    flow_pred_cond - flow_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0
                print(f"kv_cache['local_end_index']: {self.kv_cache_pos[0]['local_end_index']}")
                print(f"kv_cache['global_end_index']: {self.kv_cache_pos[0]['global_end_index']}")

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=conditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * frame_seq_length,
                cache_start=cache_origin_token,
                **temporal_loop_kwargs
            )
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=unconditional_dict,
                timestep=timestep * 0,
                kv_cache=self.kv_cache_neg,
                crossattn_cache=self.crossattn_cache_neg,
                current_start=current_start_frame * frame_seq_length,
                cache_start=cache_origin_token,
                **temporal_loop_kwargs
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        video = self.vae.decode_to_pixel(output)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        if return_latents:
            return video, output
        else:
            return video

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

    def _initialize_sample_scheduler(self, noise):
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
