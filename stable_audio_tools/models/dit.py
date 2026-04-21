import typing as tp
import math
import torch

from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .blocks import FourierFeatures
from .transformer import ContinuousTransformer

class DiffusionTransformer(nn.Module):
    def __init__(self, 
        io_channels=32, 
        patch_size=1,
        embed_dim=768,
        cond_token_dim=0,
        project_cond_tokens=True,
        global_cond_dim=0,
        project_global_cond=True,
        input_concat_dim=0,
        prepend_cond_dim=0,
        depth=12,
        num_heads=8,
        transformer_type: tp.Literal["continuous_transformer"] = "continuous_transformer",
        global_cond_type: tp.Literal["prepend", "adaLN"] = "prepend",
        timestep_cond_type: tp.Literal["global", "input_concat"] = "global",
        timestep_embed_dim=None,
        diffusion_objective: tp.Literal["v", "rectified_flow", "rf_denoiser"] = "v",
        **kwargs):

        super().__init__()
        
        self.cond_token_dim = cond_token_dim

        # Timestep embeddings
        self.timestep_cond_type = timestep_cond_type

        timestep_features_dim = 256

        self.timestep_features = FourierFeatures(1, timestep_features_dim)

        if timestep_cond_type == "global":
            timestep_embed_dim = embed_dim
        elif timestep_cond_type == "input_concat":
            assert timestep_embed_dim is not None, "timestep_embed_dim must be specified if timestep_cond_type is input_concat"
            input_concat_dim += timestep_embed_dim

        self.to_timestep_embed = nn.Sequential(
            nn.Linear(timestep_features_dim, timestep_embed_dim, bias=True),
            nn.SiLU(),
            nn.Linear(timestep_embed_dim, timestep_embed_dim, bias=True),
        )
        
        self.diffusion_objective = diffusion_objective

        if cond_token_dim > 0:
            # Conditioning tokens

            cond_embed_dim = cond_token_dim if not project_cond_tokens else embed_dim
            self.to_cond_embed = nn.Sequential(
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(cond_embed_dim, cond_embed_dim, bias=False)
            )
            
            # [NEW] Decoupled Spatial Trajectory Projection (Raw 3D Physics -> 1536)
            self.traj_to_cond_embed = nn.Sequential(
                nn.Linear(3, cond_token_dim),
                nn.SiLU(),
                nn.Linear(cond_token_dim, cond_embed_dim, bias=False)
            )
        else:
            cond_embed_dim = 0

        if global_cond_dim > 0:
            # Global conditioning
            global_embed_dim = global_cond_dim if not project_global_cond else embed_dim
            self.to_global_embed = nn.Sequential(
                nn.Linear(global_cond_dim, global_embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(global_embed_dim, global_embed_dim, bias=False)
            )

        if prepend_cond_dim > 0:
            # Prepend conditioning
            self.to_prepend_embed = nn.Sequential(
                nn.Linear(prepend_cond_dim, embed_dim, bias=False),
                nn.SiLU(),
                nn.Linear(embed_dim, embed_dim, bias=False)
            )

        self.input_concat_dim = input_concat_dim

        dim_in = io_channels + self.input_concat_dim

        self.patch_size = patch_size

        # Transformer

        self.transformer_type = transformer_type

        self.global_cond_type = global_cond_type

        if self.transformer_type == "continuous_transformer":

            global_dim = None

            if self.global_cond_type == "adaLN":
                # The global conditioning is projected to the embed_dim already at this point
                global_dim = embed_dim

            self.transformer = ContinuousTransformer(
                dim=embed_dim,
                depth=depth,
                dim_heads=embed_dim // num_heads,
                dim_in=dim_in * patch_size,
                dim_out=io_channels * patch_size,
                cross_attend = cond_token_dim > 0,
                cond_token_dim = cond_embed_dim,
                global_cond_dim=global_dim,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown transformer type: {self.transformer_type}")

        self.preprocess_conv = nn.Conv1d(dim_in, dim_in, 1, bias=False)
        nn.init.zeros_(self.preprocess_conv.weight)
        self.postprocess_conv = nn.Conv1d(io_channels, io_channels, 1, bias=False)
        nn.init.zeros_(self.postprocess_conv.weight)

    def _forward(
        self, 
        x, 
        t, 
        mask=None,
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        input_concat_cond=None,
        global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs):
        
        # [NEW] Extract Spatial Parameters and cleanup potential shadowed keys from Keywords
        spatial_trajectories = kwargs.pop("spatial_trajectories", None)
        track_times = kwargs.pop("track_times", None)
        kwargs.pop("context", None) # Remove potential shadowed text conditioning
        kwargs.pop("cross_attn_cond", None) # Remove potential shadowed cross_attn_cond
        
        _tsm_info = None

        if cross_attn_cond is not None:
            B = cross_attn_cond.shape[0]
            # Convert text semantics to 1536 implicitly expected by Transformer
            cross_attn_cond = self.to_cond_embed(cross_attn_cond) # [B, K_text*L_text, 1536]
            
            # [NEW] Robust Track Detection
            K_len_text = cross_attn_cond.shape[1]
            L_text = 128
            # K_text calculation should account for potential prepended/extra tokens
            K_text = (K_len_text + L_text - 1) // L_text
            
            if spatial_trajectories is not None:
                if spatial_trajectories.dim() == 3:
                     K_traj = 1
                     L_traj = spatial_trajectories.shape[1]
                     spatial_trajectories = spatial_trajectories.unsqueeze(1)
                else:
                     K_traj = spatial_trajectories.shape[1]
                     L_traj = spatial_trajectories.shape[2]
                
                # [NEW] Robust CFG Doubling: Handle cases where model.forward() doubled the batch for CFG 
                # but trajectories from kwargs are still at original batch size.
                if cross_attn_cond.shape[0] == 2 * spatial_trajectories.shape[0]:
                    spatial_trajectories = torch.cat([spatial_trajectories, spatial_trajectories], dim=0)
                    if track_times is not None and track_times.shape[0] == spatial_trajectories.shape[0] // 2:
                        track_times = torch.cat([track_times, track_times], dim=0)

                # Active Slicing & Linear Interpolation & Padding Engine
                T_audio = x.shape[1]
                audio_len = 8.0 # Standard Phase2 sample duration
                
                # Check for dropped tracks or diffuse backgrounds before reshaping
                is_diffuse = (spatial_trajectories.abs().sum(dim=(2, 3)) < 1e-6) # [B, K_traj]
                
                if L_traj == 0:
                    spatial_trajectories = cross_attn_cond.new_zeros(B, K_traj, T_audio, cross_attn_cond.shape[-1])
                else:
                    traj_dtype = spatial_trajectories.dtype
                    device = spatial_trajectories.device

                    start_times = spatial_trajectories.new_zeros(B, K_traj)
                    end_times = spatial_trajectories.new_full((B, K_traj), audio_len)
                    if track_times is not None:
                        num_timed_tracks = min(K_traj, track_times.shape[1])
                        if num_timed_tracks > 0:
                            track_times = track_times.to(device=device, dtype=traj_dtype)
                            start_times[:, :num_timed_tracks] = track_times[:, :num_timed_tracks, 0]
                            end_times[:, :num_timed_tracks] = track_times[:, :num_timed_tracks, 1]

                    start_times = start_times.reshape(-1)
                    end_times = end_times.reshape(-1)

                    audio_scale = T_audio / audio_len
                    raw_scale = L_traj / audio_len

                    start_ta = torch.trunc(start_times * audio_scale).to(torch.long).clamp_(0, T_audio)
                    end_ta = torch.trunc(end_times * audio_scale).to(torch.long).clamp_(0, T_audio)
                    t_active = (end_ta - start_ta).clamp_min_(1)

                    start_raw = torch.trunc(start_times * raw_scale).to(torch.long).clamp_(0, L_traj)
                    end_raw = torch.trunc(end_times * raw_scale).to(torch.long).clamp_(0, L_traj)
                    raw_len = (end_raw - start_raw).clamp_min_(1)
                    start_raw = start_raw.clamp_max_(L_traj - 1)

                    flat_trajs = spatial_trajectories.reshape(B * K_traj, L_traj, -1)
                    audio_positions = torch.arange(T_audio, device=device)
                    rel_positions = audio_positions.unsqueeze(0) - start_ta.unsqueeze(1)
                    active_mask = (rel_positions >= 0) & (rel_positions < t_active.unsqueeze(1))

                    spatial_trajectories = cross_attn_cond.new_zeros(B * K_traj, T_audio, cross_attn_cond.shape[-1])
                    track_idx, out_idx = active_mask.nonzero(as_tuple=True)

                    if track_idx.numel() > 0:
                        rel_positions = rel_positions[track_idx, out_idx].to(traj_dtype)
                        raw_len_active = raw_len[track_idx]
                        t_active_float = t_active[track_idx].to(traj_dtype)

                        source_pos = (rel_positions + 0.5) * raw_len_active.to(traj_dtype) / t_active_float - 0.5
                        left_unclamped = torch.floor(source_pos).to(torch.long)
                        weight = (source_pos - left_unclamped.to(traj_dtype)).unsqueeze(-1)

                        max_rel = raw_len_active - 1
                        zeros = torch.zeros_like(left_unclamped)
                        left = torch.minimum(torch.maximum(left_unclamped, zeros), max_rel)
                        right = torch.minimum(torch.maximum(left_unclamped + 1, zeros), max_rel)

                        raw_base = start_raw[track_idx]
                        left_vals = flat_trajs[track_idx, raw_base + left]
                        right_vals = flat_trajs[track_idx, raw_base + right]
                        active_traj = torch.lerp(left_vals, right_vals, weight)

                        active_traj_embed = self.traj_to_cond_embed(active_traj)
                        spatial_trajectories[track_idx, out_idx] = active_traj_embed.to(spatial_trajectories.dtype)

                    spatial_trajectories = spatial_trajectories.reshape(B, K_traj, T_audio, -1)
                
                # Flatten spatial_trajectories to [B, K_traj * T_audio, 1536]
                spatial_trajectories_flat = spatial_trajectories.view(B, K_traj * T_audio, -1)
                
                # Decoupled sequence concatenation (concatenate along Sequence dimension)
                cross_attn_cond = torch.cat([cross_attn_cond, spatial_trajectories_flat], dim=1)
                
                _tsm_info = {
                    "K_text": K_text,
                    "L_text": L_text,
                    "K_traj": K_traj,
                    "L_traj": T_audio, # Reassigned as T_audio correctly!
                    "track_times": track_times
                }
            else:
                _tsm_info = None

        if global_embed is not None:
            # Project the global conditioning to the embedding dimension
            global_embed = self.to_global_embed(global_embed)

        prepend_inputs = None 
        prepend_mask = None
        prepend_length = 0
        if prepend_cond is not None:
            # Project the prepend conditioning to the embedding dimension
            prepend_cond = self.to_prepend_embed(prepend_cond)

            prepend_inputs = prepend_cond
            if prepend_cond_mask is not None:
                prepend_mask = prepend_cond_mask

            prepend_length = prepend_cond.shape[1]

        if input_concat_cond is not None:
            # Interpolate input_concat_cond to the same length as x
            if input_concat_cond.shape[2] != x.shape[2]:
                input_concat_cond = F.interpolate(input_concat_cond, (x.shape[2], ), mode='nearest')

            x = torch.cat([x, input_concat_cond], dim=1)

        # Get the batch of timestep embeddings
        timestep_embed = self.to_timestep_embed(self.timestep_features(t[:, None])) # (b, embed_dim)

        # Timestep embedding is considered a global embedding. Add to the global conditioning if it exists

        if self.timestep_cond_type == "global":
            if global_embed is not None:
                global_embed = global_embed + timestep_embed
            else:
                global_embed = timestep_embed
        elif self.timestep_cond_type == "input_concat":
            x = torch.cat([x, timestep_embed.unsqueeze(1).expand(-1, -1, x.shape[2])], dim=1)

        # Add the global_embed to the prepend inputs if there is no global conditioning support in the transformer
        if self.global_cond_type == "prepend" and global_embed is not None:
            if prepend_inputs is None:
                # Prepend inputs are just the global embed, and the mask is all ones
                prepend_inputs = global_embed.unsqueeze(1)
                prepend_mask = torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)
            else:
                # Prepend inputs are the prepend conditioning + the global embed
                prepend_inputs = torch.cat([prepend_inputs, global_embed.unsqueeze(1)], dim=1)
                prepend_mask = torch.cat([prepend_mask, torch.ones((x.shape[0], 1), device=x.device, dtype=torch.bool)], dim=1)

            prepend_length = prepend_inputs.shape[1]

        x = self.preprocess_conv(x) + x

        x = rearrange(x, "b c t -> b t c")

        extra_args = {}

        if self.global_cond_type == "adaLN":
            extra_args["global_cond"] = global_embed

        if self.patch_size > 1:
            x = rearrange(x, "b (t p) c -> b t (c p)", p=self.patch_size)

        # Build TSM Attention Bias mask since we now know T_q
        if _tsm_info is not None:
            K_text = _tsm_info["K_text"]
            L_text = _tsm_info["L_text"]
            K_traj = _tsm_info["K_traj"]
            L_traj = _tsm_info["L_traj"]
            track_times = _tsm_info["track_times"]
            
            device, dtype = x.device, x.dtype
            T_audio = x.shape[1]
            num_memory = getattr(self.transformer, "num_memory_tokens", 0)
            T_q = num_memory + prepend_length + T_audio

            audio_len = 8.0 # Standard Phase2 sample duration
            
            # 1. Text Masking: Windowed/All-Pass Blocks
            K_len_text = _tsm_info.get("K_len_text", K_text * L_text)
            tsm_mask_text = torch.zeros(x.shape[0], T_audio, K_len_text, device=device, dtype=dtype)
            for b_idx in range(x.shape[0]):
                for k in range(K_text):
                    mask_k = torch.ones(T_audio, L_text, device=device, dtype=dtype) * float('-inf')
                    
                    if track_times is not None and k < track_times.shape[1]:
                        # Explicit boundaries [B, K, 2]
                        t_start = float(track_times[b_idx, k, 0].item())
                        t_end = float(track_times[b_idx, k, 1].item())
                        
                        start_ta = int((t_start / audio_len) * T_audio)
                        end_ta = int((t_end / audio_len) * T_audio)
                        
                        # Guard against out-of-bounds
                        start_ta = max(0, min(T_audio, start_ta))
                        end_ta = max(0, min(T_audio, end_ta))
                        
                        if start_ta < end_ta:
                            mask_k[start_ta:end_ta, :] = 0.0
                            
                            # [NEW] Soft Boundary Envelope (Prevent VAE spectral pop)
                            fade_len = min(5, max(1, (end_ta - start_ta) // 4))
                            if fade_len > 1:
                                fade_in = torch.linspace(-20.0, -0.01, fade_len, device=device, dtype=dtype).unsqueeze(1)
                                fade_out = torch.linspace(-0.01, -20.0, fade_len, device=device, dtype=dtype).unsqueeze(1)
                                mask_k[start_ta : start_ta+fade_len, :] += fade_in
                                mask_k[end_ta-fade_len : end_ta, :] += fade_out
                        else:
                            # If bounds collapse or invalid, fall back to all pass
                            mask_k[:, :] = 0.0
                    else:
                        mask_k[:, :] = 0.0 # No bounds provided implies all-pass
                    
                    # Slice into the actual text mask, handling tail truncation
                    c_start = k * L_text
                    c_end = min(K_len_text, (k+1) * L_text)
                    if c_start < K_len_text:
                        tsm_mask_text[b_idx, :, c_start:c_end] = mask_k[:, :c_end-c_start]
            
            # 2. Trajectory Masking: Strict 1:1 active-diagonal mapping (identity)
            tsm_mask_traj = torch.zeros(x.shape[0], T_audio, K_traj * T_audio, device=device, dtype=dtype)
            for b_idx in range(x.shape[0]):
                for k in range(K_traj):
                    if track_times is not None and k < track_times.shape[1]:
                        t_start = float(track_times[b_idx, k, 0].item())
                        t_end = float(track_times[b_idx, k, 1].item())
                    else:
                        t_start, t_end = 0.0, audio_len
                        
                    start_ta = int((t_start / audio_len) * T_audio)
                    end_ta   = int((t_end / audio_len) * T_audio)
                    start_ta = max(0, min(T_audio, start_ta))
                    end_ta = max(0, min(T_audio, end_ta))
                    
                    # 1:1 Identity
                    mask_k_track = torch.eye(T_audio, device=device, dtype=dtype)
                    
                    # [NEW] Soft Boundary Envelope for Trajectory Identity 
                    fade_len = min(5, max(1, (end_ta - start_ta) // 4))
                    if start_ta < end_ta and fade_len > 1:
                        fade_in = torch.linspace(-20.0, -0.01, fade_len, device=device, dtype=dtype)
                        fade_out = torch.linspace(-0.01, -20.0, fade_len, device=device, dtype=dtype)
                        # We subtract the fade from the identity diagonal's 1.0 positions 
                        # so that when re-mapped to 0.0, the boundaries become -20.0.
                        # Since torch.eye places 1.0 on the diagonal, we can just manipulate the diagonal!
                        diag_idx = torch.arange(T_audio, device=device)
                        mask_k_track[diag_idx[start_ta:start_ta+fade_len], diag_idx[start_ta:start_ta+fade_len]] += fade_in
                        mask_k_track[diag_idx[end_ta-fade_len:end_ta], diag_idx[end_ta-fade_len:end_ta]] += fade_out
                    
                    # Cut off tracking outside temporal envelope explicitly
                    if start_ta > 0:
                        mask_k_track[:start_ta, :] = 0.0
                    if end_ta < T_audio:
                        mask_k_track[end_ta:, :] = 0.0
                        
                    # [NEW] Dropout & Background Diffuse Detection
                    # If this track was perfectly zeroes in raw physics, it's either dropped or background.
                    if is_diffuse[b_idx, k].item():
                        mask_k_track[:, :] = 0.0
                        
                    # Mask everything else to -inf
                    mask_k_track_inf = mask_k_track.masked_fill(mask_k_track == 0.0, float('-inf'))
                    # Second, our identity elements (1.0 or 1.0+fade) get shifted down so peak is 0.0
                    # For non -inf cells, we want value to be (Original - 1.0) 
                    # If it was 1.0, it becomes 0.0. If it was 1.0 + (-20.0) it becomes -20.0 !
                    valid_mask = mask_k_track_inf != float('-inf')
                    mask_k_track_inf[valid_mask] = mask_k_track_inf[valid_mask] - 1.0
                    
                    tsm_mask_traj[b_idx, :, k*T_audio:(k+1)*T_audio] = mask_k_track_inf
            
            # 3. Concatenate Masks
            full_mask = torch.cat([tsm_mask_text, tsm_mask_traj], dim=-1) # [B, T_audio, K_len]
            
            # 4. Pad for Prepend and Memory Tokens
            if T_q > T_audio:
                pad_mask = torch.zeros(x.shape[0], T_q - T_audio, full_mask.shape[2], device=device, dtype=dtype)
                full_mask = torch.cat([pad_mask, full_mask], dim=1) # [B, T_q, K_len]
            
            cross_attn_bias = full_mask.unsqueeze(1) # [B, 1, T_q, K_len]
        
        # [NEW] Add K_len_text to _tsm_info for mask consistency
        if _tsm_info is not None:
             _tsm_info["K_len_text"] = K_len_text

        if self.transformer_type == "continuous_transformer":
            # [NEW] If we are using cross_attn_bias, we should set cross_attn_cond_mask to None 
            # to avoid length mismatch issues with the original text-only mask.
            if cross_attn_bias is not None:
                cross_attn_cond_mask = None
                
            output = self.transformer(x, prepend_embeds=prepend_inputs, context=cross_attn_cond, return_info=return_info, exit_layer_ix=exit_layer_ix, cross_attn_bias=cross_attn_bias, **extra_args, **kwargs)

            if return_info:
                output, info = output

            # Avoid postprocessing on early exit
            if exit_layer_ix is not None:
                if return_info:
                    return output, info
                else:
                    return output

        output = rearrange(output, "b t c -> b c t")[:,:,prepend_length:]

        if self.patch_size > 1:
            output = rearrange(output, "b (c p) t -> b c (t p)", p=self.patch_size)

        output = self.postprocess_conv(output) + output

        if return_info:
            return output, info

        return output

    def forward(
        self, 
        x, 
        t, 
        cross_attn_cond=None,
        cross_attn_cond_mask=None,
        negative_cross_attn_cond=None,
        negative_cross_attn_mask=None,
        input_concat_cond=None,
        global_embed=None,
        negative_global_embed=None,
        prepend_cond=None,
        prepend_cond_mask=None,
        cfg_scale=1.0,
        cfg_dropout_prob=0.0,
        cfg_interval = (0, 1),
        causal=False,
        scale_phi=0.0,
        mask=None,
        return_info=False,
        exit_layer_ix=None,
        **kwargs):

        assert causal == False, "Causal mode is not supported for DiffusionTransformer"

        model_dtype = next(self.parameters()).dtype
        
        x = x.to(model_dtype)

        t = t.to(model_dtype)

        if cross_attn_cond is not None:
            cross_attn_cond = cross_attn_cond.to(model_dtype)

        if negative_cross_attn_cond is not None:
            negative_cross_attn_cond = negative_cross_attn_cond.to(model_dtype)

        if input_concat_cond is not None:
            input_concat_cond = input_concat_cond.to(model_dtype)

        if global_embed is not None:
            global_embed = global_embed.to(model_dtype)

        if negative_global_embed is not None:
            negative_global_embed = negative_global_embed.to(model_dtype)

        if prepend_cond is not None:
            prepend_cond = prepend_cond.to(model_dtype)

        if cross_attn_cond_mask is not None:
            cross_attn_cond_mask = cross_attn_cond_mask.bool()

            cross_attn_cond_mask = None # Temporarily disabling conditioning masks due to kernel issue for flash attention

        if prepend_cond_mask is not None:
            prepend_cond_mask = prepend_cond_mask.bool()

        # Early exit bypasses CFG processing
        if exit_layer_ix is not None:
            assert self.transformer_type == "continuous_transformer", "exit_layer_ix is only supported for continuous_transformer"
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond, 
                cross_attn_cond_mask=cross_attn_cond_mask, 
                input_concat_cond=input_concat_cond, 
                global_embed=global_embed, 
                prepend_cond=prepend_cond, 
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                exit_layer_ix=exit_layer_ix,
                **kwargs
            )

        # CFG dropout
        if cfg_dropout_prob > 0.0 and cfg_scale == 1.0:
            if cross_attn_cond is not None:
                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)
                dropout_mask = torch.bernoulli(torch.full((cross_attn_cond.shape[0], 1, 1), cfg_dropout_prob, device=cross_attn_cond.device)).to(torch.bool)
                cross_attn_cond = torch.where(dropout_mask, null_embed, cross_attn_cond)

            if prepend_cond is not None:
                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)
                dropout_mask = torch.bernoulli(torch.full((prepend_cond.shape[0], 1, 1), cfg_dropout_prob, device=prepend_cond.device)).to(torch.bool)
                prepend_cond = torch.where(dropout_mask, null_embed, prepend_cond)

        if self.diffusion_objective == "v":
            sigma = torch.sin(t * math.pi / 2)
            alpha = torch.cos(t * math.pi / 2)
        elif self.diffusion_objective in ["rectified_flow", "rf_denoiser"]:
            sigma = t

        if cfg_scale != 1.0 and (cross_attn_cond is not None or prepend_cond is not None) and (cfg_interval[0] <= sigma[0] <= cfg_interval[1]):

            # Classifier-free guidance
            # Concatenate conditioned and unconditioned inputs on the batch dimension            
            batch_inputs = torch.cat([x, x], dim=0)
            batch_timestep = torch.cat([t, t], dim=0)

            if global_embed is not None:
                batch_global_cond = torch.cat([global_embed, global_embed], dim=0)
            else:
                batch_global_cond = None

            if input_concat_cond is not None:
                batch_input_concat_cond = torch.cat([input_concat_cond, input_concat_cond], dim=0)
            else:
                batch_input_concat_cond = None

            batch_cond = None
            batch_cond_masks = None
            
            # Handle CFG for cross-attention conditioning
            if cross_attn_cond is not None:

                null_embed = torch.zeros_like(cross_attn_cond, device=cross_attn_cond.device)

                # For negative cross-attention conditioning, replace the null embed with the negative cross-attention conditioning
                if negative_cross_attn_cond is not None:

                    # If there's a negative cross-attention mask, set the masked tokens to the null embed
                    if negative_cross_attn_mask is not None:
                        negative_cross_attn_mask = negative_cross_attn_mask.to(torch.bool).unsqueeze(2)

                        negative_cross_attn_cond = torch.where(negative_cross_attn_mask, negative_cross_attn_cond, null_embed)
                    
                    batch_cond = torch.cat([cross_attn_cond, negative_cross_attn_cond], dim=0)

                else:
                    batch_cond = torch.cat([cross_attn_cond, null_embed], dim=0)

                if cross_attn_cond_mask is not None:
                    batch_cond_masks = torch.cat([cross_attn_cond_mask, cross_attn_cond_mask], dim=0)
               
            batch_prepend_cond = None
            batch_prepend_cond_mask = None

            if prepend_cond is not None:

                null_embed = torch.zeros_like(prepend_cond, device=prepend_cond.device)

                batch_prepend_cond = torch.cat([prepend_cond, null_embed], dim=0)
                           
                if prepend_cond_mask is not None:
                    batch_prepend_cond_mask = torch.cat([prepend_cond_mask, prepend_cond_mask], dim=0)
         

            if mask is not None:
                batch_masks = torch.cat([mask, mask], dim=0)
            else:
                batch_masks = None
            
            batch_output = self._forward(
                batch_inputs, 
                batch_timestep, 
                cross_attn_cond=batch_cond, 
                cross_attn_cond_mask=batch_cond_masks, 
                mask = batch_masks, 
                input_concat_cond=batch_input_concat_cond, 
                global_embed = batch_global_cond,
                prepend_cond = batch_prepend_cond,
                prepend_cond_mask = batch_prepend_cond_mask,
                return_info = return_info,
                **kwargs)

            if return_info:
                batch_output, info = batch_output

            cond_output, uncond_output = torch.chunk(batch_output, 2, dim=0)

            cfg_output = uncond_output + (cond_output - uncond_output) * cfg_scale

            # CFG Rescale
            if scale_phi != 0.0:
                cond_out_std = cond_output.std(dim=1, keepdim=True)
                out_cfg_std = cfg_output.std(dim=1, keepdim=True)
                output = scale_phi * (cfg_output * (cond_out_std/out_cfg_std)) + (1-scale_phi) * cfg_output
            else:
                output = cfg_output
           
            if return_info:
                info["uncond_output"] = uncond_output
                return output, info

            return output
            
        else:
            return self._forward(
                x,
                t,
                cross_attn_cond=cross_attn_cond, 
                cross_attn_cond_mask=cross_attn_cond_mask, 
                input_concat_cond=input_concat_cond, 
                global_embed=global_embed, 
                prepend_cond=prepend_cond, 
                prepend_cond_mask=prepend_cond_mask,
                mask=mask,
                return_info=return_info,
                **kwargs
            )
