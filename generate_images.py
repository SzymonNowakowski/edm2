# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the given model."""

import os
import re
import warnings
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist



warnings.filterwarnings('ignore', '`resume_download` is deprecated')
warnings.filterwarnings('ignore', 'You are using `torch.load` with `weights_only=False`')
warnings.filterwarnings('ignore', '1Torch was not compiled with flash attention')

################################## IN WIM THE INTERNET IS UNAIVALABLE ANYWAY ##################################
############## INSTEAD, MANUALLY DOWNLOAD THE MODELS FROM THE LINKS BELOW ##############
# -----------------------------------------------------------------------------
# EDM2 model download commands
#
# mkdir -p precomputed_networks
# cd precomputed_networks
#
# === Core 512x512 FID/DINO ===
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-2147483-0.135.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-2147483-0.200.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.130.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.190.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-m-2147483-0.100.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-m-2147483-0.155.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.085.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.155.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xl-1342177-0.085.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xl-1342177-0.155.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xxl-0939524-0.070.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xxl-0939524-0.150.pkl
#
# === Core 64x64 FID ===
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.075.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-m-2147483-0.060.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-l-1073741-0.040.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-xl-0671088-0.040.pkl
#
# === Guided FID/DINO (paired net + gnet) ===
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-2147483-0.045.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.045.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-2147483-0.150.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.150.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.025.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.025.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.085.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.085.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-m-2147483-0.030.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.030.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-m-2147483-0.015.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.015.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.015.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.015.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-l-1879048-0.035.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.035.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xl-1342177-0.020.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.020.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xl-1342177-0.030.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.030.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xxl-0939524-0.015.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-2147483-0.015.pkl
#
# === Autoguided FID/DINO ===
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.070.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-0134217-0.125.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-2147483-0.120.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-0134217-0.165.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xxl-0939524-0.075.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-m-0268435-0.155.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xxl-0939524-0.130.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-m-0268435-0.205.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-uncond-2147483-0.070.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-0134217-0.110.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-s-uncond-2147483-0.090.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img512-xs-uncond-0134217-0.125.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.045.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-xs-0134217-0.110.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-s-1073741-0.105.pkl
# wget -c https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions/edm2-img64-xs-0134217-0.175.pkl
#
# === Done ===
# du -ch *.pkl | tail -n 1
#
# Total size: ~49 GB
# -----------------------------------------------------------------------------


#----------------------------------------------------------------------------
# Configuration presets.


model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm2/posthoc-reconstructions'
model_root = 'precomputed_networks/'  # Use local path where models are downloaded

config_presets = {
    'edm2-img512-xs-fid':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.135.pkl'),      # fid = 3.53
    'edm2-img512-xs-dino':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.200.pkl'),      # fd_dinov2 = 103.39
    'edm2-img512-s-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.130.pkl'),       # fid = 2.56
    'edm2-img512-s-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.190.pkl'),       # fd_dinov2 = 68.64
    'edm2-img512-m-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.100.pkl'),       # fid = 2.25
    'edm2-img512-m-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.155.pkl'),       # fd_dinov2 = 58.44
    'edm2-img512-l-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.085.pkl'),       # fid = 2.06
    'edm2-img512-l-dino':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.155.pkl'),       # fd_dinov2 = 52.25
    'edm2-img512-xl-fid':              dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.085.pkl'),      # fid = 1.96
    'edm2-img512-xl-dino':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.155.pkl'),      # fd_dinov2 = 45.96
    'edm2-img512-xxl-fid':             dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.070.pkl'),     # fid = 1.91
    'edm2-img512-xxl-dino':            dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.150.pkl'),     # fd_dinov2 = 42.84
    'edm2-img64-s-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.075.pkl'),        # fid = 1.58
    'edm2-img64-m-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-m-2147483-0.060.pkl'),        # fid = 1.43
    'edm2-img64-l-fid':                dnnlib.EasyDict(net=f'{model_root}/edm2-img64-l-1073741-0.040.pkl'),        # fid = 1.33
    'edm2-img64-xl-fid':               dnnlib.EasyDict(net=f'{model_root}/edm2-img64-xl-0671088-0.040.pkl'),       # fid = 1.33
    'edm2-img512-xs-guid-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.045.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.045.pkl', guidance=1.40), # fid = 2.91
    'edm2-img512-xs-guid-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xs-2147483-0.150.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.150.pkl', guidance=1.70), # fd_dinov2 = 79.94
    'edm2-img512-s-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.025.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.025.pkl', guidance=1.40), # fid = 2.23
    'edm2-img512-s-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.085.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.085.pkl', guidance=1.90), # fd_dinov2 = 52.32
    'edm2-img512-m-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.030.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.20), # fid = 2.01
    'edm2-img512-m-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-m-2147483-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=2.00), # fd_dinov2 = 41.98
    'edm2-img512-l-guid-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.015.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.20), # fid = 1.88
    'edm2-img512-l-guid-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-l-1879048-0.035.pkl',        gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.035.pkl', guidance=1.70), # fd_dinov2 = 38.20
    'edm2-img512-xl-guid-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.020.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.020.pkl', guidance=1.20), # fid = 1.85
    'edm2-img512-xl-guid-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xl-1342177-0.030.pkl',       gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.030.pkl', guidance=1.70), # fd_dinov2 = 35.67
    'edm2-img512-xxl-guid-fid':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.20), # fid = 1.81
    'edm2-img512-xxl-guid-dino':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.015.pkl',      gnet=f'{model_root}/edm2-img512-xs-uncond-2147483-0.015.pkl', guidance=1.70), # fd_dinov2 = 33.09
    'edm2-img512-s-autog-fid':         dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.070.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.125.pkl',        guidance=2.10), # fid = 1.34
    'edm2-img512-s-autog-dino':        dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-2147483-0.120.pkl',        gnet=f'{model_root}/edm2-img512-xs-0134217-0.165.pkl',        guidance=2.45), # fd_dinov2 = 36.67
    'edm2-img512-xxl-autog-fid':       dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.075.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.155.pkl',         guidance=2.05), # fid = 1.25
    'edm2-img512-xxl-autog-dino':      dnnlib.EasyDict(net=f'{model_root}/edm2-img512-xxl-0939524-0.130.pkl',      gnet=f'{model_root}/edm2-img512-m-0268435-0.205.pkl',         guidance=2.30), # fd_dinov2 = 24.18
    'edm2-img512-s-uncond-autog-fid':  dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.070.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.110.pkl', guidance=2.85), # fid = 3.86
    'edm2-img512-s-uncond-autog-dino': dnnlib.EasyDict(net=f'{model_root}/edm2-img512-s-uncond-2147483-0.090.pkl', gnet=f'{model_root}/edm2-img512-xs-uncond-0134217-0.125.pkl', guidance=2.90), # fd_dinov2 = 90.39
    'edm2-img64-s-autog-fid':          dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.045.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.110.pkl',         guidance=1.70), # fid = 1.01
    'edm2-img64-s-autog-dino':         dnnlib.EasyDict(net=f'{model_root}/edm2-img64-s-1073741-0.105.pkl',         gnet=f'{model_root}/edm2-img64-xs-0134217-0.175.pkl',         guidance=2.20), # fd_dinov2 = 31.85
}

#----------------------------------------------------------------------------
# EDM sampler from the paper
# "Elucidating the Design Space of Diffusion-Based Generative Models",
# extended to support classifier-free guidance.
# I extended with with alternative diffusion schedule block (which mirrors EDM template).
def edm_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=32, sigma_min=0.002, sigma_max=80, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):
    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # print all arguments
    print(f"EDM2 sampler arguments: num_steps={num_steps}, sigma_min={sigma_min}, sigma_max={sigma_max}, rho={rho}, guidane={guidance}, S_churn={S_churn}, S_min={S_min}, S_max={S_max}, S_noise={S_noise}")

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    # Create the index vector [0, 1, ..., num_steps-1] on the same device as latents, in float64.
    # We’ll use these indices to build the noise schedule.

    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    # This is the Karras (EDM and EDM2) sigma schedule.
    # It linearly interpolates between sigma_max^(1/ρ) and sigma_min^(1/ρ) and then raises back to the power ρ.
    # Result: a monotone decreasing sequence from sigma_max down to sigma_min, spaced more densely at small sigmas when ρ>1 (commonly ρ=7).

    print("The most original steps:", t_steps.detach().cpu().numpy())

    # >>>>>>>>>>>>>>>>>>>>>>> BEGIN: Alternative schedule block (mirrors EDM template) <<<<<<<<<<<<<<<<<<<<<<<<<
    # The alternative schedule replaces the segment inside [alt_sigma_min, alt_sigma_max] with a denser path.
    alt_sigma_max = 80.0          # the alternative schedule
    alt_sigma_min = 0.002
    alt_num_steps = 0        # >0 to enable the alternative schedule
    eta_divisor   = 1.0          # divide the optimal eta; use >1.0 to reduce noise below the optimal level

    if alt_num_steps > 0:
        # Build dense alt steps (descending) between alt_sigma_max and alt_sigma_min
        alt_indices = torch.linspace(0, 1, steps=alt_num_steps, dtype=dtype, device=noise.device)
        alt_steps = (alt_sigma_max ** (1.0 / rho) + alt_indices * (alt_sigma_min ** (1.0 / rho) - alt_sigma_max ** (1.0 / rho))) ** rho

        # Keep original parts outside the interval (remove from t_steps any values inside [alt_sigma_min, alt_sigma_max] range)
        above = t_steps[t_steps > alt_sigma_max]
        below = t_steps[t_steps < alt_sigma_min]
        # print("Above steps:", above.cpu().numpy())
        # print("Below steps:", below.cpu().numpy())

        # Merge everything, preserving strict descending order
        t_steps = torch.cat([above, alt_steps, below])
        print("Merged steps:", t_steps.detach().cpu().numpy())

        # Sanity check: strictly descending.
        assert torch.all(t_steps[:-1] > t_steps[1:]), "New schedule is not strictly descending!"
    else:
        print("Skipping alternative schedule; using original EDM t_steps.")

    # Append an explicit zero step (t_N = 0) for convenience
    # This is used to compute the final step size (t_{N-1} - t_N) and simplifies
    # the code below since we don't need a special case for the last step
    # It mirrors the EDM behaviour too (but in EDM they also align it with the grid, which we don't do here).
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0
    # >>>>>>>>>>>>>>>>>>>>>>> END: Alternative schedule block <<<<<<<<<<<<<<<<<<<<<<<<<

    # Main sampling loop.
    x_next = noise.to(dtype) * t_steps[0]
    # Initialize the state at the highest noise level (t_steps[0] ≈ sigma_max).
    # noise variable is expected to be standard Gaussian noise ~ N(0, I) of shape [N, C, H, W] (or whatever your model uses).
    # Multiplying by sigma_max gives a draw from N(0, sigma_max^2 I), which is the usual EDM2 starting point (pure noise).
    # It’s cast to match the integrator’s dtype.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # ===================== Alternative schedule in-loop branch (mirrors EDM + continue) =====================
        if (t_cur < alt_sigma_max) and (t_cur > alt_sigma_min):
            # iterate over pairs (t_cur, t_next); the final pair ends at exactly zero noise.
            sigma_t   = t_cur
            sigma_tm1 = t_next # next (smaller) sigma from schedule

            # gamma_tm1_reciprocal ≈ (sigma_{t+1} / sigma_t)^2, clamped to [0, 1]
            gamma_tm1_reciprocal = torch.clamp(sigma_tm1 / torch.clamp(sigma_t, min=torch.as_tensor(1e-20, dtype=dtype, device=noise.device)), max=1.0) ** 2
            # == sigma_tm1 / sigma_{t}

            # Optimal eta for the variance-preserving step, then reduce it by eta_divisor
            one = torch.as_tensor(1.0, dtype=dtype, device=noise.device)
            eta_optim_tm1 = torch.sqrt(torch.clamp(one - gamma_tm1_reciprocal, min=0.0))
            eta_used_tm1  = eta_optim_tm1 / eta_divisor

            # Coefficients for blending current state, predicted x0, and fresh noise
            square_root_tm1_s = torch.sqrt(torch.clamp(one - eta_used_tm1 ** 2, min=0.0))
            r_sqrt = torch.sqrt(torch.clamp(gamma_tm1_reciprocal, min=0.0, max=1.0))  # == (sigma_tm1 / sigma_t)
            # (alpha==1 => coef_X0 = 1 - coef_Xt)
            coef_Xt  = r_sqrt * square_root_tm1_s
            coef_X0  = one - coef_Xt
            coef_eps = sigma_tm1 * eta_used_tm1

            # EDM2 denoiser: denoise(x, t) returns ~X0 (guided if guidance != 1)
            x0_hat = denoise(x_cur, sigma_t)

            # Draw fresh noise and update.
            cur_plus_noise = coef_Xt * x_cur + coef_eps * randn_like(x_cur)
            x_next = coef_X0 * x0_hat + cur_plus_noise

            ######## Apply 2nd order (Heun) correction.
            if i < num_steps - 1: # Heun (prediction–correction) is only applied if there is another step after this.
                denoised_next = denoise(x_next, sigma_tm1)
                x_next = coef_X0 * (denoised_next + x0_hat) * 0.5 + cur_plus_noise

            continue  # Skip the standard EDM2 (churn + Heun) path on alt steps
        # ========================================================================================================
        ######################THE ORIGINAL SCHEDULE
        ####### Increase noise temporarily
        if S_churn > 0 and S_min <= t_cur <= S_max:
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1)
            # Purpose: decide how much extra noise (“churn”) to add this step.
            # S_churn/num_steps spreads the total churn over all steps.
            # It’s capped by sqrt(2)-1 ≈ 0.414 so the temporary σ can’t grow by more than ×√2.
            # Churn only happens if the current noise level t_cur is in the window [S_min, S_max]; otherwise gamma=0.

            t_hat = t_cur + gamma * t_cur
            # Compute the “churned” sigma: t_hat = (1 + gamma) * t_cur.
            # Then round it to the network’s supported σ grid (round_sigma) to match the model’s preconditioning table.

            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
            # Add just enough Gaussian noise so the total variance goes from t_cur^2 up to t_hat^2.
            # The std of the injected noise is sqrt(t_hat^2 - t_cur^2), optionally scaled by S_noise (default 1).
            # Result: x_hat has the same mean as x_cur, but a temporarily higher σ (t_hat).
        else:
            t_hat = t_cur
            x_hat = x_cur

        # Euler step.
        d_cur = (x_hat - denoise(x_hat, t_hat)) / t_hat
        # Compute the ODE slope at (x_hat, t_hat).
        # For the EDM probability-flow ODE, dx/dσ = (x - X0)/σ. Replacing X0 by denoised gives this slope.

        x_next = x_hat + (t_next - t_hat) * d_cur
        # x_next = t_next/t_hat * x_hat + (1 - t_next/t_hat) * denoised  # eqivalently
        # Explicit Euler update: move from σ = t_hat down to the scheduled next σ = t_next using slope d_cur.

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, t_next)) / t_next
            # Prediction: Re-evaluate the slope at the end of the interval (x_next, t_next).

            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
            # Heun correction (2nd order): replace the Euler result by the trapezoidal rule—average of start/end slopes times the step size, applied from the same base point x_hat.

    return x_next


def pokar_sampler(
    net, noise, labels=None, gnet=None,
    num_steps=1e4, time_min=1e-2, time_max=0.99, rho=7, guidance=1,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    dtype=torch.float32, randn_like=torch.randn_like,
):

    # Guided denoiser.
    def denoise(x, t):
        Dx = net(x, t, labels).to(dtype)
        if guidance == 1:
            return Dx
        ref_Dx = gnet(x, t, labels).to(dtype)
        return ref_Dx.lerp(Dx, guidance)

    # print all arguments
    print(f"Pokar sampler arguments: num_steps={num_steps}, sigma_min={sigma_min}, sigma_max={sigma_max}, rho={rho}, guidane={guidance}, S_churn={S_churn}, S_min={S_min}, S_max={S_max}, S_noise={S_noise}")

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=dtype, device=noise.device)
    # Create the index vector [0, 1, ..., num_steps-1] on the same device as latents, in float64.
    # We’ll use these indices to build the noise schedule.

    #karras_sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = time_min + step_indices / (num_steps - 1) * (time_max - time_min)  # time flows from time_min to time_max

    # Inverse standard normal (ppf) via torch
    z = torch.special.ndtri(t_steps)  # z = Φ^{-1}(t)

    ring_rho_inv = torch.exp(-(-1.2 + 1.2 * z))
    # in EDM schedules with alpha = 1, ring_rho = 1/sigma, so ring_rho_inv = sigma_t

    # ring_lambda_prime = 1 / pdf(qnorm(t; -1.2,1.2); -1.2,1.2)
    # For Normal(μ,σ): pdf(qnorm(t; μ,σ); μ,σ) = (1/σ) * φ(z), so 1/pdf = σ / φ(z).
    phi_z = torch.exp(-0.5 * z ** 2) / torch.sqrt(torch.tensor(2.0 * torch.pi))
    ring_lambda_prime = 1.2 / phi_z  # σ = 1.2

    F_parametrization_S_t = torch.sqrt(1 + ring_rho_inv ** 2)
    M_const = torch.max(ring_lambda_prime / F_parametrization_S_t ** 2)
    S_t_M = F_parametrization_S_t * torch.sqrt(M_const)
    lambda_prime = (S_t_M - torch.sqrt(S_t_M ** 2 - ring_lambda_prime)) ** 2
    # now we integrate numerically lambda_prime to get lambda with initial condition lambda(t0) = 0
    delta_t = (time_max - time_min) / (num_steps - 1)
    lambda_t = torch.cumsum(lambda_prime * delta_t, dim=0)
    lambda_t = lambda_t - lambda_t[0]  # remove additive constant
    rho_t = np.exp(lambda_t)  # multiplicative constant irrelevant

    r_t_inv = ring_rho_inv/rho_t

    # This is the Karras (EDM and EDM2) sigma schedule.
    # It linearly interpolates between sigma_max^(1/ρ) and sigma_min^(1/ρ) and then raises back to the power ρ.
    # Result: a monotone decreasing sequence from sigma_max down to sigma_min, spaced more densely at small sigmas when ρ>1 (commonly ρ=7).

    print("The eqivalent sigmas schedule:", ring_rho_inv.detach().cpu().numpy())
    print("The r^-1 values:", r_t_inv.detach().cpu().numpy())

    # Append an explicit final step (sigma=0) for convenience
    ring_rho_inv = torch.cat([ring_rho_inv, torch.zeros_like(ring_rho_inv[:1])])  # sigma_N = 0
    r_t_inv = torch.cat([r_t_inv, torch.zeros_like(r_t_inv[:1])])  # r^-1 = 0 at final step


    x_next = noise.to(dtype) * ring_rho_inv[0]
    # Initialize the state at the highest noise level (ring_rho_inv[0] ≈ sigma_max).
    # noise variable is expected to be standard Gaussian noise ~ N(0, I) of shape [N, C, H, W] (or whatever your model uses).
    # Multiplying by sigma_max gives a draw from N(0, sigma_max^2 I), which is the usual EDM2 starting point (pure noise).
    # It’s cast to match the integrator’s dtype.
    for i, (sigma_cur, sigma_next, r_inv_cur, r_inv_next) in enumerate(zip(ring_rho_inv[:-1], ring_rho_inv[1:], r_t_inv[:-1], r_t_inv[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        d_cur = (x_cur - denoise(x_cur, sigma_cur)) / r_inv_cur
        # Compute the ODE slope at (x_cur, sigm_cur).

        x_next = x_cur + (r_inv_next - r_inv_cur) * d_cur
        # equivalently: x_next = r_inv_next/r_inv_cur * x_cur + (1 - r_inv_next/r_inv_cur) * denoise(x_cur, sigma_cur)

        # Apply 2nd order correction.
        if i < num_steps - 1:
            d_prime = (x_next - denoise(x_next, sigma_next)) / r_inv_next
            # Prediction: Re-evaluate the slope at the end of the interval (x_next, sigma_next).

            x_next = x_next + (r_inv_next - r_inv_cur) * (0.5 * d_cur + 0.5 * d_prime)
            # Heun correction (2nd order): replace the Euler result by the trapezoidal rule—average of start/end slopes times the step size, applied from the same base point x_hat.

    return x_next


#----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Generate images for the given seeds in a distributed fashion.
# Returns an iterable that yields
# dnnlib.EasyDict(images, labels, noise, batch_idx, num_batches, indices, seeds)

def generate_images(
    net,                                        # Main network. Path, URL, or torch.nn.Module.
    gnet                = None,                 # Guiding network. None = same as main network.
    encoder             = None,                 # Instance of training.encoders.Encoder. None = load from network pickle.
    outdir              = None,                 # Where to save the output images. None = do not save.
    subdirs             = False,                # Create subdirectory for every 1000 seeds?
    seeds               = range(16, 24),        # List of random seeds.
    class_idx           = None,                 # Class label. None = select randomly.
    max_batch_size      = 32,                   # Maximum batch size for the diffusion model.
    encoder_batch_size  = 4,                    # Maximum batch size for the encoder. None = default.
    verbose             = True,                 # Enable status prints?
    device              = torch.device('cuda'), # Which compute device to use.
    sampler_fn          = edm_sampler,          # Which sampler function to use.
    **sampler_kwargs,                           # Additional arguments for the sampler function.
):
    # Rank 0 goes first.
    if dist.get_rank() != 0:
        torch.distributed.barrier()

    # Load main network.
    if isinstance(net, str):
        if verbose:
            dist.print0(f'Loading main network from {net} ...')
        with dnnlib.util.open_url(net, verbose=(verbose and dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        net = data['ema'].to(device)
        if encoder is None:
            encoder = data.get('encoder', None)
            if encoder is None:
                encoder = dnnlib.util.construct_class_by_name(class_name='training.encoders.StandardRGBEncoder')
    assert net is not None

    # Load guidance network.
    if isinstance(gnet, str):
        if verbose:
            dist.print0(f'Loading guiding network from {gnet} ...')
        with dnnlib.util.open_url(gnet, verbose=(verbose and dist.get_rank() == 0)) as f:
            gnet = pickle.load(f)['ema'].to(device)
    if gnet is None:
        gnet = net

    # Initialize encoder.
    assert encoder is not None
    if verbose:
        dist.print0(f'Setting up {type(encoder).__name__}...')
    encoder.init(device)
    if encoder_batch_size is not None and hasattr(encoder, 'batch_size'):
        encoder.batch_size = encoder_batch_size

    # Other ranks follow.
    if dist.get_rank() == 0:
        torch.distributed.barrier()

    # Divide seeds into batches.
    num_batches = max((len(seeds) - 1) // (max_batch_size * dist.get_world_size()) + 1, 1) * dist.get_world_size()
    rank_batches = np.array_split(np.arange(len(seeds)), num_batches)[dist.get_rank() :: dist.get_world_size()]
    if verbose:
        dist.print0(f'Generating {len(seeds)} images...')

    # Return an iterable over the batches.
    class ImageIterable:
        def __len__(self):
            return len(rank_batches)

        def __iter__(self):
            # Loop over batches.
            for batch_idx, indices in enumerate(rank_batches):
                r = dnnlib.EasyDict(images=None, labels=None, noise=None, batch_idx=batch_idx, num_batches=len(rank_batches), indices=indices)
                r.seeds = [seeds[idx] for idx in indices]
                if len(r.seeds) > 0:

                    # Pick noise and labels.
                    rnd = StackedRandomGenerator(device, r.seeds)
                    r.noise = rnd.randn([len(r.seeds), net.img_channels, net.img_resolution, net.img_resolution], device=device)
                    r.labels = None
                    if net.label_dim > 0:
                        r.labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[len(r.seeds)], device=device)]
                        if class_idx is not None:
                            r.labels[:, :] = 0
                            r.labels[:, class_idx] = 1

                    # Generate images.
                    latents = dnnlib.util.call_func_by_name(func_name=sampler_fn, net=net, noise=r.noise,
                        labels=r.labels, gnet=gnet, randn_like=rnd.randn_like, **sampler_kwargs)
                    r.images = encoder.decode(latents)

                    # Save images.
                    if outdir is not None:
                        for seed, image in zip(r.seeds, r.images.permute(0, 2, 3, 1).cpu().numpy()):
                            image_dir = os.path.join(outdir, f'{seed//1000*1000:06d}') if subdirs else outdir
                            os.makedirs(image_dir, exist_ok=True)
                            PIL.Image.fromarray(image, 'RGB').save(os.path.join(image_dir, f'{seed:06d}.png'))

                # Yield results.
                torch.distributed.barrier() # keep the ranks in sync
                yield r

    return ImageIterable()

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------
# Command line interface.

@click.command()
@click.option('--preset',                   help='Configuration preset', metavar='STR',                             type=str, default=None)
@click.option('--net',                      help='Main network pickle filename', metavar='PATH|URL',                type=str, default=None)
@click.option('--gnet',                     help='Guiding network pickle filename', metavar='PATH|URL',             type=str, default=None)
@click.option('--outdir',                   help='Where to save the output images', metavar='DIR',                  type=str, required=True)
@click.option('--subdirs',                  help='Create subdirectory for every 1000 seeds',                        is_flag=True)
@click.option('--seeds',                    help='List of random seeds (e.g. 1,2,5-10)', metavar='LIST',            type=parse_int_list, default='16-19', show_default=True)
@click.option('--class', 'class_idx',       help='Class label  [default: random]', metavar='INT',                   type=click.IntRange(min=0), default=None)
@click.option('--batch', 'max_batch_size',  help='Maximum batch size', metavar='INT',                               type=click.IntRange(min=1), default=32, show_default=True)

@click.option('--steps', 'num_steps',       help='Number of sampling steps', metavar='INT',                         type=click.IntRange(min=1), default=32, show_default=True)
@click.option('--sigma_min',                help='Lowest noise level', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=0.002, show_default=True)
@click.option('--sigma_max',                help='Highest noise level', metavar='FLOAT',                            type=click.FloatRange(min=0, min_open=True), default=80, show_default=True)
@click.option('--rho',                      help='Time step exponent', metavar='FLOAT',                             type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--guidance',                 help='Guidance strength  [default: 1; no guidance]', metavar='FLOAT',   type=float, default=None)
@click.option('--S_churn', 'S_churn',       help='Stochasticity strength', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_min', 'S_min',           help='Stoch. min noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default=0, show_default=True)
@click.option('--S_max', 'S_max',           help='Stoch. max noise level', metavar='FLOAT',                         type=click.FloatRange(min=0), default='inf', show_default=True)
@click.option('--S_noise', 'S_noise',       help='Stoch. noise inflation', metavar='FLOAT',                         type=float, default=1, show_default=True)
@click.option('--sampler_fn',               help="Sampler function: [default: 'edm_sampler']", metavar='STR',       type=click.Choice(['edm_sampler', 'pokar_smapler']), default='edm_sampler', show_default=True)

def cmdline(preset, **opts):
    """Generate random images using the given model.

    Examples:

    \b
    # Generate a couple of images and save them as out/*.png
    python generate_images.py --preset=edm2-img512-s-guid-dino --outdir=out

    \b
    # Generate 50000 images using 8 GPUs and save them as out/*/*.png
    torchrun --standalone --nproc_per_node=8 generate_images.py \\
        --preset=edm2-img64-s-fid --outdir=out --subdirs --seeds=0-49999
    """
    opts = dnnlib.EasyDict(opts)

    # Apply preset.
    if preset is not None:
        if preset not in config_presets:
            raise click.ClickException(f'Invalid configuration preset "{preset}"')
        for key, value in config_presets[preset].items():
            if opts[key] is None:
                opts[key] = value

    # Validate options.
    if opts.net is None:
        raise click.ClickException('Please specify either --preset or --net')
    if opts.guidance is None or opts.guidance == 1:
        opts.guidance = 1
        opts.gnet = None
    elif opts.gnet is None:
        raise click.ClickException('Please specify --gnet when using guidance')

    # Generate.
    dist.init()
    image_iter = generate_images(**opts)
    for _r in tqdm.tqdm(image_iter, unit='batch', disable=(dist.get_rank() != 0)):
        pass

#----------------------------------------------------------------------------

if __name__ == "__main__":
    cmdline()

#----------------------------------------------------------------------------
