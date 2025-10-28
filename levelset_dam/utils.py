import scipy.io
import ml_collections
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import jax.numpy as jnp
from jax.tree_util import tree_flatten
import os
import wandb
from scipy.ndimage import distance_transform_edt

def get_dataset(file_path, config):
    data = np.load(file_path, allow_pickle=True).item()
    
    phi_ref = data['phi']  # Level-set over time [t,x,y]
    u_ref = data['u']  # Velocity over time [t,x,y]
    v_ref = data['v']  # Velocity over time [t,x,y]
    p_ref = data['p']  # Pressure over time [t,x,y]
    phi0_ref = data['phi0']  # Level set field at t0
    p0_ref = data['p0']  # Pressure field at t0
    u0_ref = data['u0']  # Velocity field at t0
    v0_ref = data['v0']  # Velocity field at t0
    x_star = data['x']  # x grid points
    y_star = data['y']  # y grid points
    t_star = data['t']  # time points
    fem_mass_ape = data['fem_mass_ape'] - 100
    fem_mass_mape = fem_mass_ape.mean()

    return phi_ref, u_ref, v_ref, p_ref, phi0_ref, p0_ref, u0_ref, v0_ref, t_star, x_star, y_star, fem_mass_ape

def convert_config_to_dict(config):
    """Converts a ConfigDict object to a plain Python dictionary."""
    if isinstance(config, ml_collections.ConfigDict):
        return {k: convert_config_to_dict(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [convert_config_to_dict(v) for v in config]
    else:
        return config

def plot_collocation_points(batch, file_path):
    """Plots the collocation points for a given batch."""
    # Extract x and t coordinates
    t = batch[:, 0]
    x = batch[:, 1]

    # Plot the collocation points
    plt.figure(figsize=(8, 6))
    plt.scatter(t, x, c='blue', s=5, alpha=0.6, label='Collocation Points')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.title('Collocation Points')
    plt.grid(True)
    # plt.legend()

    # Save the figure
    plt.savefig(join(file_path,'collocation_points.png'), dpi=300)
    plt.close()

def get_bc_coords(dom, t_star, x_star, y_star):
    t0, t1 = dom[0]
    x0, x1 = dom[1]
    y0, y1 = dom[2]
    nt = t_star.shape[0]

    # u=0 on left/right (no horizontal flow through side walls)
    left_wall = jnp.stack([jnp.full_like(y_star, x0), y_star], axis=1)   # x=x0 (left)
    right_wall = jnp.stack([jnp.full_like(y_star, x1), y_star], axis=1)  # x=x1 (right)
    u_bc_coords = jnp.vstack([left_wall, right_wall])

    # v=0 on bottom/top (no vertical flow through top/bottom)
    bottom_wall = jnp.stack([x_star, jnp.full_like(x_star, y0)], axis=1)  # y=y0 (bottom)
    top_wall = jnp.stack([x_star, jnp.full_like(x_star, y1)], axis=1)     # y=y1 (top)
    v_bc_coords = jnp.vstack([bottom_wall, top_wall])

    # Replicate across time steps
    bc_coords = {}

    u_coords_tiled = jnp.tile(u_bc_coords, (nt, 1))
    time_u = jnp.repeat(t_star, u_bc_coords.shape[0])[:, None]
    bc_coords["u=0"] = jnp.hstack([time_u, u_coords_tiled])  # No horizontal flow

    v_coords_tiled = jnp.tile(v_bc_coords, (nt, 1))
    time_v = jnp.repeat(t_star, v_bc_coords.shape[0])[:, None]
    bc_coords["v=0"] = jnp.hstack([time_v, v_coords_tiled])  # No vertical flow

    p_coords_tiled = jnp.tile(top_wall, (nt, 1))
    time_p = jnp.repeat(t_star, top_wall.shape[0])[:, None]
    bc_coords["p=0"] = jnp.hstack([time_p, p_coords_tiled])  # No pressure

    return bc_coords

# def get_bc_coords_values(dom, t_star, x_star, y_star):
#     t0, t1 = dom[0]
#     x0, x1 = dom[1]
#     y0, y1 = dom[2]
    
#     # Number of time steps
#     nt = t_star.shape[0]
    
#     # Boundary walls (x,y coords)
#     left_wall = jnp.stack([jnp.full_like(y_star, x0), y_star], axis=1)    # Left boundary: x = x0
#     right_wall = jnp.stack([jnp.full_like(y_star, x1), y_star], axis=1)   # Right boundary: x = x1
#     bottom_wall = jnp.stack([x_star, jnp.full_like(x_star, y0)], axis=1)  # Bottom boundary: y = y0
#     top_wall = jnp.stack([x_star, jnp.full_like(x_star, y1)], axis=1)     # Top boundary: y = y1
#     spatial_bc_coords = jnp.vstack([left_wall, right_wall, bottom_wall, top_wall]) 
    
#     # Total number of boundary points per time step
#     n_spatial_bc = spatial_bc_coords.shape[0]
    
#     # Replicate spatial boundary coordinates across time steps
#     spatial_bc_coords_tiled = jnp.tile(spatial_bc_coords, (nt, 1))
    
#     # Create corresponding time coordinates for each spatial boundary point
#     time_bc_coords = jnp.repeat(t_star, n_spatial_bc)[:, None]
    
#     # Combine time and spatial coordinates
#     bc_coords = jnp.hstack([time_bc_coords, spatial_bc_coords_tiled])
    
#     # Create boundary values (all zeros for Dirichlet BCs)
#     bc_values = jnp.zeros((bc_coords.shape[0], 2))  # 2 outputs (u, v)

#     return bc_coords, bc_values

def count_params(state):
    """
    Calculate the total number of parameters in the model.

    Args:
        state: The TrainState object containing the model state.

    Returns:
        int: The total number of parameters.
    """
    # Extract the parameters from the state
    params = state.params

    # Flatten the parameter tree and sum the sizes of all arrays
    total_params = sum(jnp.prod(jnp.array(p.shape)) for p in tree_flatten(params)[0])

    total_bytes = 0
    
    # Flatten the parameter tree and iterate over leaf arrays
    for param in tree_flatten(params)[0]:
        total_bytes += param.size * param.dtype.itemsize

    # Convert bytes to MB
    total_mb = total_bytes / (1024 ** 2)
    return total_params, total_mb


def re_schedule_step(step, re_min, re_max, train_steps, n=5):
    """
    Computes the reynolds constant using a step function schedule.

    Parameters:
        step: The current training step (scalar).
        re_min (float): The initial constant.
        re_max (float): The final constant.
        train_steps (int): The total number of training steps.
        n (int): The number of discrete values.

    Returns:
        float: The constant at the given step.
    """
    # Ensure n is at least 1 to avoid division by zero
    if n < 1:
        raise ValueError("The number of discrete values n must be at least 1.")
    
    # Calculate the interval of steps for each g value
    interval = train_steps / n
    
    # Determine the current index based on the step
    index = jnp.floor(step / interval).astype(jnp.int32)  # Use JAX's floor and astype
    index = jnp.minimum(index, n - 1)  # Use JAX's minimum instead of Python's min
    
    # Calculate the step size for g values
    re_step = (re_max - re_min) / (n - 1) if n > 1 else 0
    
    # Compute the current g value
    re_value = re_min + index * re_step
    
    return re_value


def re_schedule_sigmoid(step, re_min, re_max, train_steps, k=10):
    """
    Computes the reynolds constant using a sigmoid-based schedule.

    Parameters:
        step: The current training step (scalar).
        re_min (float): The initial density constant.
        re_max (float): The final density constant.
        train_steps (int): The total number of training steps.
        k (float): The steepness of the sigmoid curve. Higher values make the transition sharper.

    Returns:
        float: The constant at the given step.
    """
    # Normalize the step to the range [0, 1]
    t = step / train_steps

    # Compute the sigmoid transition
    sigmoid = 1 / (1 + jnp.exp(-k * (t - 0.5)))

    # Scale the sigmoid output to the range [rho_min, rho_max]
    re_value = re_min + (re_max - re_min) * sigmoid

    return re_value


def plot_re_schedule(config, workdir, schedule_func, re_max):
    """
    Plots the evolution of re through training.
    """
    steps = np.arange(config.training.max_steps + 1)
    if schedule_func == "step":
        re_values = [re_schedule_step(step, config.training.re_min, re_max, config.training.max_steps, n=5) for step in steps]
    elif schedule_func == "sigmoid":
        re_values = [re_schedule_sigmoid(step, config.training.re_min, re_max, config.training.max_steps, k=10) for step in steps]
    
    plt.figure(figsize=(8, 5))
    plt.plot(steps, re_values, label=schedule_func)
    plt.xlabel("Training Steps")
    plt.ylabel("Reynolds")
    plt.title("Evolution of Reynolds during training")
    plt.legend()
    plt.grid()

    # Show the plot
    # Save dir
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir,'re_evolution.png'), dpi=300)

    wandb.log({f"re evolution": wandb.Image(os.path.join(save_dir, 're_evolution.png'))})


def geometric_sdf_reinit(phi, dx, dy):
    """Reinitialize the level set function using the geometric SDF method."""
    mask_pos = phi >= 0
    mask_neg = phi < 0
    sdf_pos = distance_transform_edt(mask_pos, sampling=(dx, dy))
    sdf_neg = distance_transform_edt(mask_neg, sampling=(dx, dy))
    return sdf_pos - sdf_neg


def save_phi0(phi0, save_dir, file_name, idx, reinit):
    """Plot and save phi0 as a PNG image and log to wandb."""
    plt.figure(figsize=(6, 4))
    plt.pcolormesh(phi0.T, cmap='jet', shading='flat')
    plt.colorbar(label=r"$\phi_0$")
    plt.title(f"Level Set φ₀ at Window {idx} | {reinit} reinit")
    plt.xlabel("x index")
    plt.ylabel("y index")
    plt.contour(phi0.T, levels=[0.01], colors='black', linewidths=1.5)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    full_path = os.path.join(save_dir, file_name)
    plt.savefig(full_path, dpi=150, bbox_inches='tight')
    plt.close()

    wandb.log({f"phi0 window {idx}": wandb.Image(full_path)})