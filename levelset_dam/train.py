import os, sys
import time
import re
import gc
from functools import partial

import numpy as np
import scipy

import jax
import jax.numpy as jnp
from jax import random, vmap, pmap, debug
from jax.tree_util import tree_map
from flax import jax_utils

import scipy.io

import ml_collections
from absl import logging
import wandb
import platform

from jaxpi.archs import PeriodEmbs, Embedding
from jaxpi.samplers import UniformSampler, TimeSpaceSampler, BaseSampler
from jaxpi.logging import Logger
from jaxpi.utils import save_checkpoint, restore_checkpoint

import models
from utils import get_dataset, convert_config_to_dict, plot_collocation_points, get_bc_coords, count_params, re_schedule_sigmoid, re_schedule_step, geometric_sdf_reinit, save_phi0


def train_one_window(config, workdir, model, res_sampler, idx, phi_star, p_star, u_star, v_star, multi=None):
    # Initialize logger
    logger = Logger()

    # Initialize evaluator
    evaluator = models.LevelSetEvaluator(config, model)

    step_offset = idx * config.training.max_steps

    # Scale steps for first time window
    num_steps = config.training.max_steps
    if idx == 0 and config.training.num_time_windows > 1:
        num_steps = config.training.max_steps * config.training.first_s2s_steps_factor
        step_offset = idx * config.training.max_steps
        logger.info(f"Scaling steps for first time window by {config.training.first_s2s_steps_factor} | steps: {num_steps}")
    elif idx > 0 and config.training.num_time_windows > 1:
        step_offset = idx * config.training.max_steps + config.training.max_steps * (config.training.first_s2s_steps_factor - 1)  # Offset by the scaled steps
        logger.info(f"Using step offset for time window {idx + 1}: {step_offset}")
    else:
        num_steps = config.training.max_steps

    logger.info("Waiting for JIT...")
    start_time = time.time()
    for step in range(num_steps):
        batch = next(res_sampler)
        # if step == 0:
        #     save_dir = os.path.join(workdir,'figures',config.wandb.name)
        #     if not os.path.isdir(save_dir):
        #         os.makedirs(save_dir)
        #     plot_collocation_points(np.array(batch).reshape(-1,2),save_dir)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)
                # Keep eik_p constant
                # model.state.weights["eik_p"] = np.array([config.weighting.init_weights["eik_p"]], dtype=np.float32)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, phi_star, p_star, u_star, v_star)

                # Get global step if using curriculum training
                if config.transfer.curriculum == True:
                    step_logged = step + config.logging.global_step
                    
                    # Skip last step except for last dataset in curriculum
                    if (step+1) != config.training.max_steps:
                        wandb.log(log_dict, step_logged + step_offset)
                        end_time = time.time()
                        logger.log_iter(step_logged, start_time, end_time, log_dict)

                # With multi level
                elif multi == True:
                    log_step = (config.training.num_time_windows * config.training1.max_steps0) + step + step_offset
                    print(f"log step: {log_step}")
                    wandb.log(log_dict, log_step)
                    end_time = time.time()
                    logger.log_iter(step, start_time, end_time, log_dict)

                # Regular
                else:
                    wandb.log(log_dict, step + step_offset)
                    end_time = time.time()
                    logger.log_iter(step, start_time, end_time, log_dict)
                start_time = end_time

        # Saving
        if config.saving.save_every_steps is not None:
            if (step + 1) % config.saving.save_every_steps == 0 or (
                step + 1
            ) == config.training.max_steps:
                if multi == True:
                    ckpt_path = os.path.join(os.getcwd(), "checkpoints", config.wandb.name, "ckpt", "time_window_{}_level_1".format(idx + 1))
                else:
                    ckpt_path = os.path.join(os.getcwd(), "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
                save_checkpoint(model.state, ckpt_path, keep=config.saving.num_keep_ckpts)

    return model

def train_and_evaluate_s2s(config: ml_collections.ConfigDict, workdir: str):
    """Train and evaluate the model using sequence 2 sequence learning from Krishnapriyan et al. (2021)"""
    # Initialize logger
    logger = Logger()

    # Find out if running on pc for dubugging or on HPC without internet access
    if 'microsoft' in platform.uname().release.lower():
        mode = "online"
    else:
        mode = "offline"
        
    # Initialize W&B
    wandb_config = config.wandb
    wandb.init(project=wandb_config.project, name=wandb_config.name, mode=mode, config=convert_config_to_dict(config), tags=wandb_config.tag, notes=wandb_config.notes)
    logger.info(f"wandb initialized {mode}")
    logger.info(f"Project: {config.wandb.project}")
    logger.info(f"Config name: {config.wandb.name}")
    logger.info(f"Notes: {config.wandb.notes}")
    logger.info(f"Optimizer: {config.optim.optimizer}")
    logger.info(f"Dataset: {config.dataset}")
    logger.info(f"Re: {config.nondim.Re}")
    logger.info(f"rho_ratio: {config.nondim.rho_ratio}")
    logger.info(f"mu_ratio: {config.nondim.mu_ratio}")
    logger.info(f"nu_ratio: {config.mu2/config.rho2}/{config.mu1/config.rho1}")

    # Get dataset
    phi_ref, u_ref, v_ref, p_ref, phi0_ref, p0_ref, u0_ref, v0_ref, t_ref, x_ref, y_ref, fem_mass_ape = get_dataset(os.path.join("data",config.dataset), config)

    if config.nondim.nondimensionalize == True:
        t_star = t_ref / config.nondim.T_star
        x_star = x_ref / config.nondim.X_star
        y_star = y_ref / config.nondim.Y_star
        phi_star = phi_ref / config.nondim.PHI_star
        p_star = p_ref / config.nondim.P_star
        u_star = u_ref / config.nondim.U_star
        v_star = v_ref / config.nondim.V_star
        phi0_star = phi0_ref / config.nondim.PHI_star
        p0_star = p0_ref / config.nondim.P_star
        u0_star = u0_ref / config.nondim.U_star
        v0_star = v0_ref / config.nondim.V_star
    else:
        t_star = t_ref
        x_star = x_ref
        y_star = y_ref
        phi_star = phi_ref
        p_star = p_ref
        u_star = u_ref
        v_star = v_ref
        phi0_star = phi0_ref
        p0_star = p0_ref
        u0_star = u0_ref
        v0_star = v0_ref

    # phi0 = phi_star[0, :, :]
    phi0 = phi0_star
    p0 = p0_star
    u0 = u0_star
    v0 = v0_star

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    dx = x_star[1] - x_star[0]
    dy = y_star[1] - y_star[0]
    # phi_dx = config.nondim.PHI_star / len(x_star)
    # phi_dy = config.nondim.PHI_star / len(y_star)

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    t0 = t[0]
    dt = round(t[-1] - t[-2],6)
    t1 = t[-1] + dt # * (1 + 0.01)  # cover the end point of each time window

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    # Define bc coords
    bc_coords = get_bc_coords(dom, t, x_star, y_star)

    keys = random.split(random.PRNGKey(0), config.training.num_time_windows)

    # Define residual sampler of collocation points
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # Get temporal coords for initial time window
    # temp_coords = temporal_coords[0 : num_time_steps]

    # Precompute re values for gradual training
    if config.training.re_schedule != None:
        re_values = np.zeros(config.training.max_steps)
        for step in range(config.training.max_steps):

            if config.training.re_schedule == "step":
                re_values[step] = re_schedule_step(step, config.training.re_min, config.nondim.Re, config.training.max_steps, config.training.re_schedule_n)

            elif config.training.re_schedule == "sigmoid":
                re_values[step] = re_schedule_sigmoid(step, config.training.re_min, config.nondim.Re, config.training.max_steps, config.training.re_schedule_k)
        re_values = jnp.array(re_values)
    else:
        re_values = None

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))

        # Get the reference solution for the current time window
        phi = phi_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        u = u_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        v = v_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Physics-informed initialization
        if config.use_pi_init == True and config.transfer.s2s_pi_init == True:
            logger.info("Use physics-informed initialization...")

            model = models.LevelSet(config, phi0, p0, u0, v0, t, x_star, y_star, bc_coords, re_values)
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            params = state.params

            # Initialization data source
            if config.pi_init_type == "initial_condition":
                t_init = t_star[::10]
                x = x_star
                y = y_star
                phi_init = phi0
                p_init = p0
                u_init = u0
                v_init = v0
                # debug.print("p0: {p0}",p0=p0)

                t_scaled = t_init / t_init[-1]

                tt, xx, yy = jnp.meshgrid(t_scaled, x, y, indexing="ij")
                inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None], yy.flatten()[:, None]])
                phi_init = jnp.tile(phi_init, (t_scaled.shape[0], 1, 1))
                p_init = jnp.tile(p_init, (t_scaled.shape[0], 1, 1))
                u_init = jnp.tile(u_init, (t_scaled.shape[0], 1, 1))
                v_init = jnp.tile(v_init, (t_scaled.shape[0], 1, 1))

            feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

            phi_coeffs, phi_res, rank, s = jnp.linalg.lstsq(feat_matrix, phi_init.flatten(), rcond=None)
            p_coeffs, p_res, rank, s = jnp.linalg.lstsq(feat_matrix, p_init.flatten(), rcond=None)
            u_coeffs, u_res, rank, s = jnp.linalg.lstsq(feat_matrix, u_init.flatten(), rcond=None)
            v_coeffs, v_res, rank, s = jnp.linalg.lstsq(feat_matrix, v_init.flatten(), rcond=None)

            logger.info(f"least square phi residuals: {phi_res}")
            logger.info(f"least square p residuals: {p_res}")
            logger.info(f"least square u residuals: {u_res}")
            logger.info(f"least square v residuals: {v_res}")

            coeffs = jnp.vstack([phi_coeffs, p_coeffs, u_coeffs, v_coeffs]).T
            config.arch.pi_init = coeffs

            del model, state, params

        # Calculate time offset to account for time reset to [0,t] on each time window, for vx,vy calculation
        time_offset = idx * t1

        # Initialize model
        model = models.LevelSet(config, phi0, p0, u0, v0, t, x_star, y_star, bc_coords, re_values, time_offset=time_offset)

        # Count params
        total_params, total_mb = count_params(model.state)
        logger.info(f"Amount of params: {total_params}")
        logger.info(f"Model size: {round(total_mb,3)} mb")

        # Transfer params between time windows for init for s2s and curriculum purposes
        if config.transfer.s2s_transfer == True and config.transfer.s2s_pi_init == False:
            logger.info(f"About to restore model from checkpoint")
            if config.transfer.curriculum == True and idx == 0:
                ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_1")
            else:
                ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx))
            if os.path.exists(ckpt_path):
                    state = restore_checkpoint(model.state, ckpt_path, step=None) # Load latest checkpoint for tw
                    state = jax_utils.replicate(state)

                    if config.training.s2s_transfer_weights == True:
                        model.state = model.state.replace(params=state.params, weights=state.weights)
                    else:
                        model.state = model.state.replace(params=state.params)

        # Train model for the current time window
        model = train_one_window(config, workdir, model, res_sampler, idx, phi, p, u, v)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1 and config.training.num_time_windows != (idx+1):
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params
            phi0 = model.phi0_pred_fn(params, t_star[num_time_steps], x_star, y_star)
            p0 = model.p0_pred_fn(params, t_star[num_time_steps], x_star, y_star)
            u0 = model.u0_pred_fn(params, t_star[num_time_steps], x_star, y_star)
            v0 = model.v0_pred_fn(params, t_star[num_time_steps], x_star, y_star)

            # del model, state, params

        # Reinit phi0 for next window
        if config.training.reinit == True and config.training.num_time_windows > 1:
            save_dir = os.path.join(workdir, "figures", config.wandb.name)
            save_phi0(phi0, save_dir, f'phi0_idx{idx}_before.png', idx, 'before')

            # Refine level set for reinit
            len_coarse = len(x_star)
            if len_coarse % 2 == 0:
                len_fine = config.training.reinit_refine_scale * len_coarse
            else:
                len_fine = config.training.reinit_refine_scale * (len_coarse - 1) + 1
            x_star_fine = np.linspace(x_star[0], x_star[-1], len_fine)
            y_star_fine = np.linspace(y_star[0], y_star[-1], len_fine)
            dx_fine = x_star_fine[1] - x_star_fine[0]
            dy_fine = y_star_fine[1] - y_star_fine[0]
            phi0 = model.phi0_pred_fn(params, t_star[num_time_steps], x_star_fine, y_star_fine)
            
            # Reinit level set with fine grid
            phi0 = geometric_sdf_reinit(phi0, dx_fine*config.nondim.X_star, dy_fine*config.nondim.X_star)  # Fast EDT method

            # Downsample back to regular grid
            phi0 = phi0[::config.training.reinit_refine_scale, ::config.training.reinit_refine_scale]
            
            save_phi0(phi0, save_dir, f'phi0_idx{idx}_after.png', idx, 'after')


        # Reset if we use pi_init
        if idx == 0:
            config.transfer.s2s_pi_init = False # Turn off after first time window