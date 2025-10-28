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
from utils import get_dataset, convert_config_to_dict, plot_collocation_points, get_bc_coords_values, count_params


def train_one_window(config, workdir, model, res_sampler, idx, p_ref, multi=None):
    # Initialize logger
    logger = Logger()

    # Initialize evaluator
    evaluator = models.LevelSetEvaluator(config, model)

    step_offset = idx * config.training.max_steps

    logger.info("Waiting for JIT...")
    start_time = time.time()
    for step in range(config.training.max_steps):
        batch = next(res_sampler)
        model.state = model.step(model.state, batch)

        if config.weighting.scheme in ["grad_norm", "ntk"]:
            if step % config.weighting.update_every_steps == 0:
                model.state = model.update_weights(model.state, batch)

        # Log training metrics, only use host 0 to record results
        if jax.process_index() == 0:
            if step % config.logging.log_every_steps == 0:
                # Get the first replica of the state and batch
                state = jax.device_get(tree_map(lambda x: x[0], model.state))
                batch = jax.device_get(tree_map(lambda x: x[0], batch))
                log_dict = evaluator(state, batch, p_ref)

                # Get global step if using curriculum training
                if config.transfer.curriculum == True:
                    step_logged = step + config.logging.global_step
                    
                    # Skip last step except for last dataset in curriculum
                    if (step+1) != config.training.max_steps:
                        wandb.log(log_dict, step_logged + step_offset)
                        end_time = time.time()
                        logger.log_iter(step_logged, start_time, end_time, log_dict)

                # Without curriculum
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

    # Get dataset
    p_ref, t_ref, x_ref, y_ref, p0, fem_mass_ape = get_dataset(os.path.join("data",config.dataset))

    if config.nondim.nondimensionalize == True:
        t_star = t_ref / config.nondim.T_star
        x_star = x_ref / config.nondim.X_star
        y_star = y_ref / config.nondim.Y_star
        p_star = p_ref / config.nondim.P_star
    else:
        t_star = t_ref
        x_star = x_ref
        y_star = y_ref
        p_star = p_ref

    # p0 = p_star[0, :, :]

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    t0 = t[0]
    dt = round(t[-1] - t[-2],6)
    t1 = t[-1] + dt # * (1 + 0.01)  # cover the end point of each time window

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    # Create a mask for points inside the circular region
    X, Y = np.meshgrid(x_star, y_star, indexing='ij')
    center = np.array([0.5, 0.5])  # Center of the circular region
    radius = 0.45  # Radius of the circular region
    distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = distance <= radius

    keys = random.split(random.PRNGKey(0), config.training.num_time_windows)

    # Define residual sampler of collocation points
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))


    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))

        # Get the reference solution for the current time window
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Physics-informed initialization
        if config.use_pi_init == True and config.transfer.s2s_pi_init == True:
            logger.info("Use physics-informed initialization...")

            model = models.LevelSet(config, p0, t, x_star, y_star, mask=mask)
            state = jax.device_get(tree_map(lambda x: x[0], model.state))
            params = state.params

            # Initialization data source
            if config.pi_init_type == "linear_pde":
                # load data
                data = np.load(os.path.join("data",config.dataset), allow_pickle=True).item()
                # downsample the grid and data
                p_init = data['level_set_data'][::10]
                t_init = data["t"].flatten()[::10]
                x = data["x"].flatten()
                y = data["y"].flatten()
                
                t_scaled = t_init / t_init[-1]

                tt, xx, yy = jnp.meshgrid(t_scaled, x, y, indexing="ij")
                inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None], yy.flatten()[:, None]])

            elif config.pi_init_type == "initial_condition":
                t_init = t_star[::10]
                x = x_star
                y = y_star
                p_init = p0

                t_scaled = t_init / t_init[-1]

                tt, xx, yy = jnp.meshgrid(t_scaled, x, y, indexing="ij")
                inputs = jnp.hstack([tt.flatten()[:, None], xx.flatten()[:, None], yy.flatten()[:, None]])
                p_init = jnp.tile(p_init, (t_scaled.shape[0], 1, 1))

            feat_matrix, _ = vmap(state.apply_fn, (None, 0))(params, inputs)

            p_coeffs, p_res, rank, s = jnp.linalg.lstsq(feat_matrix, p_init.flatten(), rcond=None)

            logger.info(f"least square p residuals: {p_res}")

            coeffs = jnp.vstack([p_coeffs]).T
            config.arch.pi_init = coeffs

            del model, state, params

        # Calculate time offset to account for time reset to [0,t] on each time window, for vx,vy calculation
        time_offset = idx * t1

        # Initialize model
        model = models.LevelSet(config, p0, t, x_star, y_star, time_offset=time_offset, mask=mask)

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

                    model.state = model.state.replace(params=state.params, weights=state.weights)

        # Train model for the current time window
        model = train_one_window(config, workdir, model, res_sampler, idx, p)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params
            p0 = model.p0_pred_fn(params, t_star[num_time_steps * (idx+1)], x_star, y_star)

            del model, state, params

        # Reset if we use pi_init
        if idx == 0:
            config.transfer.s2s_pi_init = False # Turn off after first time window


def train_two_level(config: ml_collections.ConfigDict, workdir: str):
    # Initialize logger
    logger = Logger()
    logger.info(f"Training level 0...")

    # Find out if running on pc for dubugging or on HPC without internet access
    if 'microsoft' in platform.uname().release.lower():
        mode = "online"
    else:
        mode = "offline"
        
    # Initialize W&B !!! Comment out!!!
    # wandb_config = config.wandb
    # wandb.init(project=wandb_config.project, name=wandb_config.name, mode=mode, config=convert_config_to_dict(config), tags=wandb_config.tag, notes=wandb_config.notes)
    # logger.info(f"wandb initialized {mode}")

    # Start by training the first level
    train_and_evaluate_s2s(config, workdir)

    # Get dataset
    p_ref, t_ref, x_ref, y_ref, p0, fem_mass_ape = get_dataset(os.path.join("data",config.dataset))

    if config.nondim.nondimensionalize == True:
        t_star = t_ref / config.nondim.T_star
        x_star = x_ref / config.nondim.X_star
        y_star = y_ref / config.nondim.Y_star
        p_star = p_ref / config.nondim.P_star
    else:
        t_star = t_ref
        x_star = x_ref
        y_star = y_ref
        p_star = p_ref

    x0 = x_star[0]
    x1 = x_star[-1]

    y0 = y_star[0]
    y1 = y_star[-1]

    # Get the time domain for each time window
    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    t0 = t[0]
    dt = round(t[-1] - t[-2],6)
    t1 = t[-1] + dt # * (1 + 0.01)  # cover the end point of each time window

    # Define domain
    dom = jnp.array([[t0, t1], [x0, x1], [y0, y1]])

    # Create a mask for points inside the circular region
    X, Y = np.meshgrid(x_star, y_star, indexing='ij')
    center = np.array([config.mask.center_x, config.mask.center_y])  # Center of the circular region
    radius = config.mask.radius  # Radius of the circular region
    distance = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = distance <= radius

    # Define residual sampler of collocation points
    res_sampler = iter(UniformSampler(dom, config.training1.batch_size_per_device))

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))

        # Get the reference solution for the current time window
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Calculate time offset to account for time reset to [0,t] on each time window, for vx,vy calculation
        time_offset = idx * t1

        # To restore model
        if config.use_pi_init:
            config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

        # Initialize model
        model = models.LevelSetMulti(config, p0, t, x_star, y_star, time_offset=time_offset, mask=mask)

        # Restore model from level 0
        logger.info(f"Restoring model from level 0 training...")
        ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx+1))
        if os.path.exists(ckpt_path):
            state = restore_checkpoint(model.state, ckpt_path, step=None) # Load latest checkpoint for tw

            # Add an additional array embedding to every element
            def add_array_embedding(x):
                return jnp.array([x])
            
            params = {'params':None}
            params['params'] = jax.tree_map(add_array_embedding, state.params['params'])
            # weights = jax.tree_map(add_array_embedding, state.weights)
            weights = {key: jnp.array([value]) for key, value in config.weighting1.init_weights.items()}
            opt_state = jax.tree_map(add_array_embedding, state.opt_state)
            step = jax.tree_map(add_array_embedding, state.step)
            momentum = jax.tree_map(add_array_embedding, state.momentum)
            model.state = model.state.replace(params=params, weights=weights, step=step, opt_state=opt_state, momentum=momentum)

            # Deactivate useless features in second level
            config.weighting.use_causal = False

            # Update weighting and training features for second level
            config.weighting.scheme = config.weighting1.scheme
            config.weighting.init_weights = config.weighting1.init_weights
            config.weighting.momentum = config.weighting1.momentum
            config.weighting.update_every_steps = config.weighting1.update_every_steps
            config.training.max_steps = config.training1.max_steps
            config.training.batch_size_per_device = config.training1.batch_size_per_device
            config.logging.log_every_steps = config.training1.log_every_steps

        # Train model for the current time window
        model = train_one_window(config, workdir, model, res_sampler, idx, p, multi=True)

        # Update the initial condition for the next time window
        if config.training.num_time_windows > 1:
            state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model.state))
            params = state.params
            p0 = model.p0_pred_fn(params, t_star[num_time_steps * (idx+1)], x_star, y_star)

            del model, state, params