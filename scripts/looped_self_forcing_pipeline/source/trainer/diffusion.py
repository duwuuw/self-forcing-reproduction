import gc
import logging
import re
from pathlib import Path

from model import CausalDiffusion
from model.lora import build_lora_checkpoint_payload, prepare_generator_for_lora
from utils.dataset import ShardingLMDBDataset, cycle
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

from utils.distributed import (
    EMA_FSDP,
    barrier,
    fsdp_lora_state_dict,
    fsdp_wrap,
    launch_distributed_job,
    load_fsdp_lora_state_dict,
    load_optimizer_state_dict_for_checkpoint,
    optimizer_state_dict_for_checkpoint,
)


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer
        self.model = CausalDiffusion(config, device=self.device)
        self.model.load_generator_base_checkpoint(config.generator_ckpt)
        self.model.lora_parameter_names = prepare_generator_for_lora(
            self.model.generator.model,
            self.model.temporal_loop,
            self.model.lora_config,
        )
        self._broadcast_lora_parameters()
        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            min_num_params=40_000_000,
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy
        )

        if not config.no_visualize or config.load_raw_video:
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        dataset = ShardingLMDBDataset(config.data_path, max_pair=int(1e8))
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        self.generator_ema = None
        self._maybe_initialize_generator_ema()

        resume_checkpoint = getattr(config, "resume_checkpoint", None)
        if resume_checkpoint:
            self._load_resume_checkpoint(resume_checkpoint)

        self.max_grad_norm = 10.0
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        ema_enabled = (
            self.config.ema_weight is not None and self.config.ema_weight > 0.0
        )
        if (
            ema_enabled
            and self.step >= self.config.ema_start_step
            and self.generator_ema is None
        ):
            raise RuntimeError(
                "EMA should be initialized before saving this training step"
            )

        state_dict = build_lora_checkpoint_payload(
            generator=fsdp_lora_state_dict(
                self.model.generator,
                self.model.lora_parameter_names,
            ),
            critic=None,
            generator_ema=(
                self.generator_ema.state_dict()
                if self.generator_ema is not None else None
            ),
            generator_optimizer=optimizer_state_dict_for_checkpoint(
                self.model.generator,
                self.generator_optimizer,
            ),
            critic_optimizer=None,
            metadata=self._lora_checkpoint_metadata(),
        )

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def _lora_checkpoint_metadata(self):
        loop = self.model.temporal_loop
        student_model = getattr(self.config.model_kwargs, "model_name", None)
        if student_model is None:
            student_model = self.config.model_kwargs.get("model_name")
        lora_config = (
            OmegaConf.to_container(self.config.lora, resolve=True)
            if OmegaConf.is_config(self.config.lora)
            else dict(self.config.lora)
        )
        return {
            "training_objective": "supervised_flow_matching",
            "step": self.step,
            "seed": int(self.config.seed),
            "student_model": student_model,
            "base_checkpoint": str(
                Path(self.config.generator_ckpt).expanduser().resolve(strict=True)
            ),
            "temporal_loop": {
                "enabled": loop.enabled,
                "mode": loop.mode,
                "layer_start": loop.layer_start,
                "layer_end": loop.layer_end,
                "k_min": loop.k_min,
                "k_max": loop.k_max,
                "strength": loop.strength,
                "stop_grad_early": loop.stop_grad_early,
                "schedule": loop.schedule,
                "runtime_num_layers": loop.runtime_num_layers,
                "training_enabled": loop.training_enabled,
            },
            "lora": lora_config,
            "ema": {
                "enabled": (
                    self.config.ema_weight is not None
                    and self.config.ema_weight > 0.0
                ),
                "decay": self.config.ema_weight,
                "start_step": self.config.ema_start_step,
            },
        }

    def _maybe_initialize_generator_ema(self):
        ema_weight = self.config.ema_weight
        if (
            self.generator_ema is not None
            or ema_weight is None
            or ema_weight <= 0.0
            or self.step < self.config.ema_start_step
        ):
            return

        print(f"Setting up EMA with weight {ema_weight}")
        self.generator_ema = EMA_FSDP(
            self.model.generator,
            decay=ema_weight,
            parameter_filter=lambda name, parameter: (
                "lora_" in name and parameter.requires_grad
            ),
        )

    def _load_resume_checkpoint(self, checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict):
            raise ValueError("resume checkpoint must contain a mapping")
        if payload.get("generator_format") != "lora_adapter":
            raise ValueError("resume_checkpoint requires an adapter-only generator")

        metadata = payload.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("training_objective") != "supervised_flow_matching"
        ):
            raise ValueError(
                "resume_checkpoint requires supervised flow-matching metadata"
            )
        self._validate_resume_compatibility(metadata)
        if payload.get("critic") is not None or payload.get("critic_optimizer") is not None:
            raise ValueError("supervised resume checkpoint must not contain critic state")
        generator_state = payload.get("generator")
        if not isinstance(generator_state, dict) or not generator_state:
            raise ValueError("resume checkpoint is missing generator adapter state")

        step = metadata.get("step")
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError(
                "resume checkpoint metadata.step must be a non-negative integer"
            )
        if payload.get("generator_optimizer") is None:
            raise ValueError("resume checkpoint is missing generator optimizer state")
        ema_state = payload.get("generator_ema")
        ema_enabled = (
            self.config.ema_weight is not None and self.config.ema_weight > 0.0
        )
        if (
            ema_enabled
            and step >= self.config.ema_start_step
            and (not isinstance(ema_state, dict) or not ema_state)
        ):
            raise ValueError(
                "resume checkpoint is missing EMA state for an active EMA configuration"
            )
        if ema_enabled and step < self.config.ema_start_step and ema_state is not None:
            raise ValueError("resume checkpoint has EMA state before ema_start_step")
        if not ema_enabled and ema_state is not None:
            raise ValueError("resume checkpoint has EMA state but EMA is disabled")

        ema_template = None
        if ema_enabled and step >= self.config.ema_start_step:
            ema_template = self.generator_ema
            if ema_template is None:
                ema_template = EMA_FSDP(
                    self.model.generator,
                    decay=self.config.ema_weight,
                    parameter_filter=lambda name, parameter: (
                        "lora_" in name and parameter.requires_grad
                    ),
                )
            expected_ema = ema_template.state_dict()
            if set(ema_state) != set(expected_ema):
                raise ValueError(
                    "resume checkpoint EMA adapter scope does not match model"
                )
            for name, expected in expected_ema.items():
                value = ema_state[name]
                if not isinstance(value, torch.Tensor):
                    raise ValueError(
                        f"resume checkpoint EMA value is not a tensor: {name}"
                    )
                if value.shape != expected.shape or value.dtype != expected.dtype:
                    raise ValueError(
                        f"resume checkpoint EMA tensor shape or dtype does not match: {name}"
                    )

        load_fsdp_lora_state_dict(self.model.generator, generator_state)
        load_optimizer_state_dict_for_checkpoint(
            self.model.generator,
            self.generator_optimizer,
            payload["generator_optimizer"],
        )

        self.step = step
        self.generator_ema = ema_template
        if self.generator_ema is not None and ema_state is not None:
            self.generator_ema.load_state_dict(ema_state)

    def _validate_resume_compatibility(self, metadata):
        expected = self._lora_checkpoint_metadata()
        for key in (
            "training_objective",
            "student_model",
            "temporal_loop",
            "lora",
            "ema",
        ):
            if metadata.get(key) != expected[key]:
                raise ValueError(f"resume checkpoint {key} does not match training config")

        relative_base = os.environ.get(
            "SUE_BASE_CHECKPOINT_RELATIVE_PATH", ""
        ).strip()
        expected_hash = os.environ.get("SUE_BASE_CHECKPOINT_SHA256", "").strip()
        if relative_base or expected_hash:
            self._validate_sue_base_checkpoint_identity(
                metadata,
                relative_base=relative_base,
                expected_hash=expected_hash,
            )
            return

        saved_base = metadata.get("base_checkpoint")
        if not isinstance(saved_base, str) or not Path(saved_base).is_absolute():
            raise ValueError(
                "resume checkpoint base path must be absolute outside SUE runtime"
            )
        try:
            saved_path = Path(saved_base).expanduser().resolve(strict=True)
            active_path = Path(self.config.generator_ckpt).expanduser().resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError("resume checkpoint base path cannot be resolved") from error
        if saved_path != active_path:
            raise ValueError("resume checkpoint base path does not match active generator")

    def _validate_sue_base_checkpoint_identity(
        self, metadata, *, relative_base: str, expected_hash: str
    ):
        if not relative_base or not expected_hash:
            raise ValueError("SUE base checkpoint identity requires both path and SHA-256")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
            raise ValueError("SUE base checkpoint SHA-256 is invalid")

        relative = Path(relative_base).expanduser()
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("SUE base checkpoint path must be relative to SUE_ASSET_ROOT")
        asset_root_value = os.environ.get("SUE_ASSET_ROOT", "").strip()
        if not asset_root_value:
            raise ValueError("SUE_ASSET_ROOT is required to validate resume base identity")
        try:
            asset_root = Path(asset_root_value).expanduser().resolve(strict=True)
            active_path = Path(self.config.generator_ckpt).expanduser().resolve(strict=True)
            resolved_asset = (asset_root / relative).resolve(strict=True)
        except (OSError, RuntimeError, ValueError) as error:
            raise ValueError("SUE base checkpoint path cannot be resolved") from error

        if not resolved_asset.is_file() or resolved_asset != active_path:
            raise ValueError("SUE base checkpoint path does not match active generator")
        if metadata.get("base_checkpoint") != relative.as_posix():
            raise ValueError("resume checkpoint SUE base reference does not match runtime")
        saved_hash = metadata.get("base_checkpoint_sha256")
        if (
            not isinstance(saved_hash, str)
            or not re.fullmatch(r"[0-9a-fA-F]{64}", saved_hash)
            or saved_hash.lower() != expected_hash.lower()
        ):
            raise ValueError("resume checkpoint base SHA-256 does not match runtime")

    def train_one_step(self, batch):
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if "clean_latent" in batch:
            clean_latent = batch["clean_latent"].to(
                device=self.device, dtype=self.dtype
            )
        elif not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["ode_latent"][:, -1].to(
                device=self.device, dtype=self.dtype)
        else:  # encode raw video to latent
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)
            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
        # Step 2: Extract the conditional infos
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            conditional_dict=conditional_dict,
            clean_latent=clean_latent,
        )
        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        if self.generator_ema is not None:
            self.generator_ema.update(self.model.generator)

        # Increment the step since we finished gradient update
        self.step += 1
        self._maybe_initialize_generator_ema()

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # Step 4: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    def _broadcast_lora_parameters(self):
        if not dist.is_available() or not dist.is_initialized():
            return
        if torch.cuda.is_available():
            sync_device = torch.device("cuda", torch.cuda.current_device())
        else:
            sync_device = torch.device("cpu")
        parameters = dict(self.model.generator.model.named_parameters())
        with torch.no_grad():
            for name in self.model.lora_parameter_names:
                if name not in parameters:
                    raise ValueError(f"injected LoRA parameter is missing: {name}")
                parameter = parameters[name]
                synchronized = parameter.detach().to(device=sync_device)
                dist.broadcast(synchronized, src=0)
                parameter.copy_(synchronized.to(device=parameter.device))

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        sampled_noise = torch.randn(
            [batch_size, 21, 16, 60, 104], device="cuda", dtype=self.dtype
        )
        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def train(self):
        while True:
            batch = next(self.dataloader)
            self.train_one_step(batch)
            if (not self.config.no_save) and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
