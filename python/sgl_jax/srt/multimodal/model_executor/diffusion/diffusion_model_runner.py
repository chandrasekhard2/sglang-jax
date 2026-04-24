import logging
import time
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx
from jax import NamedSharding
from jax.sharding import PartitionSpec
from tqdm import tqdm

from sgl_jax.srt.configs.load_config import LoadConfig
from sgl_jax.srt.model_executor.base_model_runner import BaseModelRunner
from sgl_jax.srt.model_loader.loader import JAXModelLoader, get_model_loader
from sgl_jax.srt.multimodal.common.ServerArgs import MultimodalServerArgs
from sgl_jax.srt.multimodal.configs.config_registry import get_diffusion_config
from sgl_jax.srt.multimodal.manager.schedule_batch import Req
from sgl_jax.srt.multimodal.models.diffusion_solvers.flow_unipc_multistep_scheduler import (
    FlowUniPCMultistepScheduler,
)
from sgl_jax.srt.utils.jax_utils import device_array

logger = logging.getLogger(__name__)


# DiffusionModelRunner is responsible for running denoising steps within diffusion model inference
class DiffusionModelRunner(BaseModelRunner):
    def __init__(
        self,
        server_args: MultimodalServerArgs,
        mesh: jax.sharding.Mesh = None,
        model_class=None,
        stage_sub_dir: str | None = None,
    ):
        self.server_args = server_args
        self.mesh = mesh
        load_sub_dir = "transformer" if stage_sub_dir is None else stage_sub_dir
        if load_sub_dir == "":
            load_sub_dir = None
        self.load_sub_dir = load_sub_dir
        load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            sub_dir=load_sub_dir,
        )
        self.model_loader = JAXModelLoader(load_config, mesh)

        self.transformer_model = None
        self.transformer_2_model = None
        self.solver = None
        self.guidance = None
        self._cache_dit_enabled = False
        self._cached_num_steps = None
        self.model_class = model_class
        # TODO: load model_config from server_args based on model architecture
        self.model_config = get_diffusion_config(self.server_args.model_path)
        self.model_config.model_path = self.server_args.model_path
        self.model_config.model_class = self.model_class
        # Additional initialization for diffusion model if needed
        # e.g., setting up noise schedulers, diffusion steps, etc.
        self.initialize()

    def initialize(self):
        # self.model = self.model_loader.load_model(model_config=self.model_config)
        load_sub_dir = self.load_sub_dir
        self.model_loader = get_model_loader(
            mesh=self.mesh, load_config=LoadConfig(sub_dir=load_sub_dir)
        )
        self.model = self.model_loader.load_model(model_config=self.model_config)
        use_dynamic_shifting = getattr(self.model_config, "use_dynamic_shifting", False)
        scheduler_type = getattr(self.model_config, "scheduler_type", "FlowUniPCMultistepScheduler")
        
        if scheduler_type == "EulerScheduler":
            from sgl_jax.srt.multimodal.models.diffusion_solvers.euler_scheduler import EulerScheduler
            base_shift = getattr(self.model_config, "base_shift", 0.95)
            max_shift = getattr(self.model_config, "max_shift", 2.05)
            self.solver = EulerScheduler(
                base_shift=base_shift,
                max_shift=max_shift,
                stretch=True,
                terminal=0.1,
            )
        else:
            self.solver = FlowUniPCMultistepScheduler(
                shift=self.model_config.flow_shift if not use_dynamic_shifting else None,
                use_dynamic_shifting=use_dynamic_shifting,
            )
        # self.solver_state = self.solver.create_state()
        # Any additional initialization specific to diffusion models
        self.initialize_jit()

    def initialize_jit(self):
        model_def, model_state = nnx.split(self.model)
        model_state_leaves, model_state_def = jax.tree_util.tree_flatten(model_state)

        @partial(
            jax.jit,
            static_argnames=["model_state_def"],
        )
        def forward_model(
            model_def,
            model_state_def,
            model_state_leaves,
            hidden_states,
            encoder_hidden_states,
            timesteps,
            encoder_hidden_states_image,
            guidance_scale,
            audio_latent=None,
            audio_context=None,
        ):
            model_state = jax.tree_util.tree_unflatten(model_state_def, model_state_leaves)
            model = nnx.merge(model_def, model_state)
            return model(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timesteps=timesteps,
                encoder_hidden_states_image=encoder_hidden_states_image,
                guidance_scale=guidance_scale,
                audio_latent=audio_latent,
                audio_context=audio_context,
            )

        def forward_wrapper(
            hidden_states,
            encoder_hidden_states,
            timesteps,
            encoder_hidden_states_image,
            guidance_scale,
            audio_latent=None,
            audio_context=None,
        ):
            return forward_model(
                model_def,
                model_state_def,
                model_state_leaves,
                hidden_states,
                encoder_hidden_states,
                timesteps,
                encoder_hidden_states_image,
                guidance_scale,
                audio_latent=audio_latent,
                audio_context=audio_context,
            )

        self.jitted_forward = forward_wrapper

    def forward(
        self,
        batch: Req,
        mesh: jax.sharding.Mesh,
        abort_checker: Callable[[], bool] | None = None,
        step_callback: Callable[[], None] | None = None,
    ) -> bool:
        """Run diffusion inference with optional abort checking.

        Args:
            batch: Request batch containing embeddings and parameters.
            mesh: JAX device mesh for sharding.
            abort_checker: Optional callback that returns True if the request
                should be aborted. Called between diffusion steps.
            step_callback: Optional callback invoked after each denoising step
                completes, used for profiling step counting.

        Returns:
            True if the request was aborted, False otherwise.
        """
        num_inference_steps = batch.num_inference_steps
        guidance_scale = batch.guidance_scale
        stg_scale = getattr(batch, "stg_scale", 1.0)
        do_spatio_temporal_guidance = stg_scale > 0.0 and getattr(self.model_config, "stg_mode", False)
        do_classifier_free_guidance = guidance_scale > 1.0 and not do_spatio_temporal_guidance

        # Handle prompt embeddings

        prompt_embeds = batch.prompt_embeds
        # Add batch dimension if needed: (L, D) -> (1, L, D)
        if prompt_embeds.ndim == 2:
            prompt_embeds = jnp.expand_dims(prompt_embeds, axis=0)

        # Pad to 512 tokens (do this first for both positive and negative)
        def pad_to_512(embeds):
            # LTX2 does not pad to 512, Wan does. We pad if configured.
            pad_dim = getattr(self.model_config, "max_sequence_length", 512)
            if embeds.shape[1] < pad_dim:
                pad_width = pad_dim - embeds.shape[1]
                return jnp.pad(
                    embeds, ((0, 0), (0, pad_width), (0, 0)), mode="constant", constant_values=0
                )
            return embeds

        # text_embeds shape: (B, max_len, D)
        prompt_embeds = pad_to_512(prompt_embeds)
        if do_spatio_temporal_guidance:
            if batch.negative_prompt_embeds is not None:
                neg_embeds = batch.negative_prompt_embeds
                if neg_embeds.ndim == 2:
                    neg_embeds = jnp.expand_dims(neg_embeds, axis=0)
                neg_embeds = pad_to_512(neg_embeds)
                # STG concatenates [pos, neg, pos]
                prompt_embeds = jnp.concatenate([prompt_embeds, neg_embeds, prompt_embeds], axis=0)
        elif do_classifier_free_guidance:
            if batch.negative_prompt_embeds is not None:
                neg_embeds = batch.negative_prompt_embeds
                # Add batch dimension to negative embeds if needed
                if neg_embeds.ndim == 2:
                    neg_embeds = jnp.expand_dims(neg_embeds, axis=0)

                # Pad negative embeds to 512 as well
                neg_embeds = pad_to_512(neg_embeds)

                # Now both are (1, max_len, D), concatenate to (2, max_len, D)
                prompt_embeds = jnp.concatenate([prompt_embeds, neg_embeds], axis=0)
            else:
                pass

        text_embeds = device_array(
            prompt_embeds, sharding=NamedSharding(self.mesh, PartitionSpec())
        )
        
        audio_context = None
        if getattr(self.model_config, "is_audio_enabled", False) and getattr(batch, "audio_prompt_embeds", None) is not None:
            audio_prompt_embeds = batch.audio_prompt_embeds
            if audio_prompt_embeds.ndim == 2:
                audio_prompt_embeds = jnp.expand_dims(audio_prompt_embeds, axis=0)
            audio_prompt_embeds = pad_to_512(audio_prompt_embeds)
            
            if do_spatio_temporal_guidance:
                if getattr(batch, "audio_negative_prompt_embeds", None) is not None:
                    audio_neg_embeds = batch.audio_negative_prompt_embeds
                    if audio_neg_embeds.ndim == 2:
                        audio_neg_embeds = jnp.expand_dims(audio_neg_embeds, axis=0)
                    audio_neg_embeds = pad_to_512(audio_neg_embeds)
                    audio_prompt_embeds = jnp.concatenate([audio_prompt_embeds, audio_neg_embeds, audio_prompt_embeds], axis=0)
            elif do_classifier_free_guidance:
                if getattr(batch, "audio_negative_prompt_embeds", None) is not None:
                    audio_neg_embeds = batch.audio_negative_prompt_embeds
                    if audio_neg_embeds.ndim == 2:
                        audio_neg_embeds = jnp.expand_dims(audio_neg_embeds, axis=0)
                    audio_neg_embeds = pad_to_512(audio_neg_embeds)
                    audio_prompt_embeds = jnp.concatenate([audio_prompt_embeds, audio_neg_embeds], axis=0)
                    
            audio_context = device_array(
                audio_prompt_embeds, sharding=NamedSharding(self.mesh, PartitionSpec())
            )

        self.prepare_latents(batch)
        latents = device_array(batch.latents, sharding=NamedSharding(self.mesh, PartitionSpec()))
        audio_latents = None
        if getattr(self.model_config, "is_audio_enabled", False) and getattr(batch, "audio_latents", None) is not None:
            audio_latents = device_array(batch.audio_latents, sharding=NamedSharding(self.mesh, PartitionSpec()))
        
        # Calculate mu for dynamic shifting if needed
        mu = None
        if getattr(self.model_config, "use_dynamic_shifting", False):
            # LTX-2 token calculation: T * H * W
            shape = latents.transpose(0, 4, 1, 2, 3).shape
            tokens = shape[2] * shape[3] * shape[4]
            base_shift = getattr(self.model_config, "base_shift", 0.95)
            max_shift = getattr(self.model_config, "max_shift", 2.05)
            mm = (max_shift - base_shift) / (4096 - 1024)
            b = base_shift - mm * 1024
            mu = tokens * mm + b

        self.solver.set_timesteps(
            num_inference_steps=num_inference_steps,
            shape=latents.transpose(0, 4, 1, 2, 3).shape,
            mu=mu,
        )
        self.solver.set_begin_index(0)
        start_time = time.time()
        import jax._src.test_util as jtu

        for step in tqdm(range(num_inference_steps), desc="Diffusion steps"):
            # Check for abort between steps
            if abort_checker is not None and abort_checker():
                logger.info(
                    "Diffusion aborted at step %d/%d for rid=%s",
                    step,
                    num_inference_steps,
                    batch.rid,
                )
                return True  # Aborted

            jax.profiler.StepTraceAnnotation("diffusion_step", step_num=step)
            t_scalar = jnp.array(self.solver.timesteps, dtype=jnp.int32)[step]
            if do_spatio_temporal_guidance:
                latents_in = jnp.concatenate([latents] * 3, axis=0)
                audio_latents_in = jnp.concatenate([audio_latents] * 3, axis=0) if audio_latents is not None else None
            elif do_classifier_free_guidance:
                latents_in = jnp.concatenate([latents] * 2, axis=0)
                audio_latents_in = jnp.concatenate([audio_latents] * 2, axis=0) if audio_latents is not None else None
            else:
                latents_in = latents
                audio_latents_in = audio_latents
            # Create timestep batch AFTER latents concat to match batch size
            t_batch = jnp.broadcast_to(t_scalar, (latents_in.shape[0],))
            # Transpose to channel-first (B, T, H, W, C) -> (B, C, T, H, W) for model
            latents_cf = latents_in.transpose(0, 4, 1, 2, 3)
            # Perform denoising step
            with jtu.count_pjit_cpp_cache_miss() as count:
                noise_pred_out = self.jitted_forward(
                    hidden_states=latents_cf,
                    encoder_hidden_states=text_embeds,
                    timesteps=t_batch,
                    encoder_hidden_states_image=None,
                    guidance_scale=None,
                    audio_latent=audio_latents_in,
                    audio_context=audio_context,
                )
                if count() > 0:
                    logger.info("diffusion cache miss count: %d", count())

            # Handle multimodal outputs (dict with "video" and "audio" keys)
            is_multimodal = isinstance(noise_pred_out, dict)
            noise_pred = noise_pred_out["video"] if is_multimodal else noise_pred_out
            audio_noise_pred = noise_pred_out["audio"] if is_multimodal else None

            if do_spatio_temporal_guidance:
                bsz = latents_in.shape[0] // 3
                v_cond, v_uncond, v_ptb = noise_pred[:bsz], noise_pred[bsz:2*bsz], noise_pred[2*bsz:]

                # Flow matching x0 prediction
                sigma = float(self.solver._sigmas[step])
                latents_slice = latents_cf[:bsz]

                x0_cond = latents_slice - v_cond * sigma
                x0_uncond = latents_slice - v_uncond * sigma
                x0_ptb = latents_slice - v_ptb * sigma
                x0_pred = x0_cond + (guidance_scale - 1.0) * (x0_cond - x0_uncond) + stg_scale * (x0_cond - x0_ptb)

                rescale_scale = getattr(batch, "rescale_scale", 0.7)
                factor = jnp.std(x0_cond, ddof=1) / jnp.maximum(jnp.std(x0_pred, ddof=1), 1e-6)
                factor = rescale_scale * factor + (1.0 - rescale_scale)
                x0_pred *= factor

                noise_pred = (latents_slice - x0_pred) / sigma
                
                if audio_noise_pred is not None and audio_latents is not None:
                    audio_guidance_scale = getattr(batch, "audio_guidance_scale", 7.0)
                    audio_stg_scale = getattr(batch, "audio_stg_scale", 1.0)
                    a_cond, a_uncond, a_ptb = audio_noise_pred[:bsz], audio_noise_pred[bsz:2*bsz], audio_noise_pred[2*bsz:]
                    audio_latents_slice = audio_latents[:bsz]
                    
                    a_x0_cond = audio_latents_slice - a_cond * sigma
                    a_x0_uncond = audio_latents_slice - a_uncond * sigma
                    a_x0_ptb = audio_latents_slice - a_ptb * sigma
                    a_x0_pred = a_x0_cond + (audio_guidance_scale - 1.0) * (a_x0_cond - a_x0_uncond) + audio_stg_scale * (a_x0_cond - a_x0_ptb)
                    
                    a_factor = jnp.std(a_x0_cond, ddof=1) / jnp.maximum(jnp.std(a_x0_pred, ddof=1), 1e-6)
                    a_factor = rescale_scale * a_factor + (1.0 - rescale_scale)
                    a_x0_pred *= a_factor
                    
                    audio_noise_pred = (audio_latents_slice - a_x0_pred) / sigma
                    
            elif do_classifier_free_guidance:
                bsz = latents_in.shape[0] // 2
                noise_uncond = noise_pred[bsz:]
                noise_pred = noise_pred[:bsz]
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)
                
                if audio_noise_pred is not None:
                    audio_guidance_scale = getattr(batch, "audio_guidance_scale", 7.0)
                    a_noise_uncond = audio_noise_pred[bsz:]
                    audio_noise_pred = audio_noise_pred[:bsz]
                    audio_noise_pred = a_noise_uncond + audio_guidance_scale * (audio_noise_pred - a_noise_uncond)

            # noise_pred is already channel-first (B, C, T, H, W) from model
            # latents is channel-last (B, T, H, W, C), need to transpose for solver
            latents = self.solver.step(
                model_output=noise_pred,  # already (B, C, T, H, W)
                timestep=t_scalar,
                sample=latents.transpose(0, 4, 1, 2, 3),  # (B, T, H, W, C) -> (B, C, T, H, W)
                return_dict=False,
            )[0]
            latents = latents.transpose(0, 2, 3, 4, 1)  # back to channel-last
            
            # Step audio
            if audio_noise_pred is not None and audio_latents is not None:
                # Euler solver manual inline calculation to avoid index mutations
                sigma = float(self.solver._sigmas[step])
                sigma_next = float(self.solver._sigmas[step + 1])
                dt = sigma_next - sigma
                
                sample_fp32 = audio_latents.astype(jnp.float32)
                velocity_fp32 = audio_noise_pred.astype(jnp.float32)
                
                prev_sample = sample_fp32 + velocity_fp32 * dt
                audio_latents = prev_sample.astype(audio_latents.dtype)

            if step_callback is not None:
                step_callback()

        logger.info("Finished diffusion step %d in %.2f seconds", step, time.time() - start_time)
        batch.latents = jax.device_get(latents)
        if getattr(self.model_config, "is_audio_enabled", False) and audio_latents is not None:
            batch.audio_latents = jax.device_get(audio_latents)
        return False  # Not aborted

    def prepare_latents(self, batch: Req):
        if batch.latents is not None:
            return
        assert batch.width % self.model_config.scale_factor_spatial == 0
        assert batch.height % self.model_config.scale_factor_spatial == 0
        if batch.num_frames is not None:
            assert (batch.num_frames - 1) % self.model_config.scale_factor_temporal == 0
        latents = jax.random.normal(
            jax.random.PRNGKey(46),
            (
                1,
                (
                    (batch.num_frames - 1) // self.model_config.scale_factor_temporal + 1
                    if batch.num_frames is not None
                    else 1
                ),
                batch.height // self.model_config.scale_factor_spatial,
                batch.width // self.model_config.scale_factor_spatial,
                self.model_config.latent_input_dim,
            ),
            dtype=jnp.float32,
        )  # Placeholder for latents
        batch.latents = latents
        
        if getattr(self.model_config, "is_audio_enabled", False):
            fps = getattr(self.model_config, "fps", 24.0)
            duration = (batch.num_frames) / fps
            latents_per_second = 25.0
            audio_frames = int(round(duration * latents_per_second))
            audio_dim = getattr(self.model_config, "audio_in_channels", 128)
            batch.audio_latents = jax.random.normal(
                jax.random.PRNGKey(47),
                (1, audio_frames, audio_dim),
                dtype=jnp.float32,
            )
