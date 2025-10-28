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


class ResSpaceTimeSampler(BaseSampler):
    """Space-time sampler used when the values of a given space-time point varies over time.
    The sampler acts as a lookup table that will look through the temporal_coords and return the coordinate values for a given time step.
    
    temporal_coords: lookup table with shape (num_time_steps, num_spatial_points, n coordinate information (t,x,y,vx,vy,...))"""
    def __init__(
        self, temporal_coords, batch_size, rng_key=random.PRNGKey(1234)
    ):
        super().__init__(batch_size, rng_key)
        self.temporal_coords = temporal_coords    # Shape: (num_time_steps, num_spatial_points, 5 (t,x,y,vx,vy))
        self.num_time_steps = temporal_coords.shape[0]      # Number of time steps (21)
        self.num_spatial_points = temporal_coords.shape[1]

    @partial(pmap, static_broadcasted_argnums=(0,))
    def data_generation(self, key):
        "Generates data containing batch_size samples"
        key1, key2 = random.split(key)

        # Sample batch_size time indices uniformly between 0 and num_time_steps-1
        time_indices = random.randint(
            key1,
            shape=(self.batch_size,),  # (32,)
            minval=0,
            maxval=self.num_time_steps
        )

        # Sample spatial indices randomly
        spatial_idx = random.choice(
            key2, self.num_spatial_points, shape=(self.batch_size,)
        )   # (32,)

        # Retrieve corresponding spatial data for the sampled time indices
        # `temporal_coords` shape: (num_time_steps, num_spatial_points, 5)
        batch = self.temporal_coords[time_indices, spatial_idx, :]    # (32, 5)

        return batch



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
                log_dict = evaluator(state, batch, p_ref)

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

    # Get dataset
    p_ref, t_ref, x_ref, y_ref, fem_mass_ape = get_dataset(os.path.join("data",config.dataset))

    if config.nondim.nondimensionalize == True:
        t_star = t_ref / config.nondim.T_star
        x_star = x_ref / config.nondim.X_star
        y_star = y_ref / config.nondim.Y_star
        p_star = p_ref / config.nondim.P_star
        # p_star = (p_ref - jnp.mean(p_ref)) / jnp.std(p_ref)
    else:
        t_star = t_ref
        x_star = x_ref
        y_star = y_ref
        p_star = p_ref

    p0 = p_star[0, :, :]

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

    # Define coords
    # temporal_coords = jnp.array(temporal_coords)    # (21,441,4) (n time steps, n points, x/y/vx/vy)

    # Define bc coords
    # bc_coords, bc_values = get_bc_coords_values(dom, t_star, x_star, y_star)

    # Get absolute vx (constant velocity in x)
    if config.dataset.split('_')[2] == "cst":
        config.vx = float(re.search(r'vx([0-9]*\.[0-9]+|[0-9]+)', config.dataset).group(1))
    elif config.dataset.split('_')[0] == "vortex":
        # config.vx = float(re.search(r'vx([0-9.]+)_T([0-9.]+)', config.dataset).group(1))
        config.T  = float(re.search(r'T([0-9]+)', config.dataset).group(1))
    elif config.dataset.split('_')[2] == "linear":
        config.vx = float(re.search(r'vx([-\d.]+)_vy([-\d.]+)_T([0-9]+)\.npy', config.dataset).group(1))
        config.vy = float(re.search(r'vx([-\d.]+)_vy([-\d.]+)_T([0-9]+)\.npy', config.dataset).group(2))
        config.T  = float(re.search(r'vx([-\d.]+)_vy([-\d.]+)_T([0-9]+)\.npy', config.dataset).group(3))

    keys = random.split(random.PRNGKey(0), config.training.num_time_windows)

    # Define residual sampler of collocation points
    res_sampler = iter(UniformSampler(dom, config.training.batch_size_per_device))

    # Get temporal coords for initial time window
    # temp_coords = temporal_coords[0 : num_time_steps]

    for idx in range(config.training.num_time_windows):
        logging.info("Training time window {}".format(idx + 1))

        # res_sampler = iter(
        #         ResSpaceTimeSampler(
        #             temp_coords,
        #             config.training.batch_size_per_device,
        #             rng_key=keys[idx],
        #         )
        # )

        # Get the reference solution for the current time window
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        # Physics-informed initialization
        if config.use_pi_init == True and config.transfer.s2s_pi_init == True:
            logger.info("Use physics-informed initialization...")

            model = models.LevelSet(config, p0, t, x_star, y_star)
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
                # debug.print("p0: {p0}",p0=p0)
                # np.save(os.path.join("..","testing","test_plots",f"p0_{idx}.npy"),p0)

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
        model = models.LevelSet(config, p0, t, x_star, y_star, time_offset=time_offset)

        # Count params
        total_params, total_mb = count_params(model.state)
        logger.info(f"Amount of params: {total_params}")
        logger.info(f"Model size: {round(total_mb,3)} mb")

        # # Update initial weights and params if transfer learning
        # if idx == 0:
        #     if config.transfer.curriculum == True:
        #         ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt")
        #         if os.path.exists(os.path.join(ckpt_path,f"checkpoint_{config.transfer.curri_step}")):
        #             state = restore_checkpoint(model.state, ckpt_path, step=config.transfer.curri_step)

        #             # Add an additional array embedding to every element
        #             def add_array_embedding(x):
        #                 return jnp.array([x])
                    
        #             params = {'params':None}
        #             params['params'] = jax.tree_map(add_array_embedding, state.params['params'])
        #             weights = jax.tree_map(add_array_embedding, state.weights)
        #             model.state = model.state.replace(params=params, weights=weights)

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
            # np.save(os.path.join("..","testing","test_plots",f"p0_{idx}_pred.npy"),p0)
            # p0 = p_star[num_time_steps * (idx+1)]
            # np.save(os.path.join("..","testing","test_plots",f"p0_{idx}_ref.npy"),p0)
            # debug.print("p0: {p0}",p0=p0)
            # p_pred = model.p_pred_fn(params, t, x_star, y_star)
            # p0 = p_pred[-1]
            # debug.print("p0: {p0}",p0=p0)

            # Reset time coordinate for length of time window
            # temp_coords = temporal_coords[(idx+1)*num_time_steps : (idx+2)*num_time_steps]
            # temp_coords = np.array(temp_coords)
            # temp_coords[:,:,0] = temp_coords[:,:,0] - (t1 * (idx+1)) + dt
            # temp_coords = jnp.array(temp_coords)

            del model, state, params

        # Reset if we use pi_init
        if idx == 0:
            config.transfer.s2s_pi_init = False # Turn off after first time window
        # mid_time_window = int(np.ceil(config.training.num_time_windows / 2))
        # if (idx+1) == mid_time_window:
        #     config.transfer.s2s_pi_init = True # Turn on to reset the gradients when reversing the flow
        # else:
        #     config.transfer.s2s_pi_init = False



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
    p_ref, t_ref, x_ref, y_ref, fem_mass_ape = get_dataset(os.path.join("data",config.dataset))

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

    p0 = p_star[0, :, :]

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
        model = models.LevelSetMulti(config, p0, t, x_star, y_star, time_offset=time_offset)

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
            if idx == 0:
                config.training1.max_steps0 = config.training.max_steps
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