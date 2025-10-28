import scipy.io
import ml_collections
import matplotlib.pyplot as plt
import numpy as np
from os.path import join
import jax.numpy as jnp
from jax.tree_util import tree_flatten

def get_dataset(file_path):
    data = np.load(file_path, allow_pickle=True).item()
    
    phi_ref = data['phi']  # Level-set function over time [t,x,y] (41,101,101)
    x_star = data['x']  # x grid points (101,)
    y_star = data['y']  # y grid points (101,)
    t_star = data['t']  # time points (41,)
    p0 = data['p0_sq']  # phi0 with a unit square grid (101,101)
    fem_mass_ape = data['fem_mass_ape'] - 100 # (41,)

    return phi_ref, t_star, x_star, y_star, p0, fem_mass_ape


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

def get_bc_coords_values(dom, t_star, x_star, y_star):
    t0, t1 = dom[0]
    x0, x1 = dom[1]
    y0, y1 = dom[2]
    
    # Number of time steps
    nt = t_star.shape[0]
    
    # Boundary walls (x,y coords)
    left_wall = jnp.stack([jnp.full_like(y_star, x0), y_star], axis=1)    # Left boundary: x = x0
    right_wall = jnp.stack([jnp.full_like(y_star, x1), y_star], axis=1)   # Right boundary: x = x1
    bottom_wall = jnp.stack([x_star, jnp.full_like(x_star, y0)], axis=1)  # Bottom boundary: y = y0
    top_wall = jnp.stack([x_star, jnp.full_like(x_star, y1)], axis=1)     # Top boundary: y = y1
    spatial_bc_coords = jnp.vstack([left_wall, right_wall, bottom_wall, top_wall]) 
    
    # Total number of boundary points per time step
    n_spatial_bc = spatial_bc_coords.shape[0]
    
    # Replicate spatial boundary coordinates across time steps
    spatial_bc_coords_tiled = jnp.tile(spatial_bc_coords, (nt, 1))
    
    # Create corresponding time coordinates for each spatial boundary point
    time_bc_coords = jnp.repeat(t_star, n_spatial_bc)[:, None]
    
    # Combine time and spatial coordinates
    bc_coords = jnp.hstack([time_bc_coords, spatial_bc_coords_tiled])
    
    # Create boundary values (all zeros for Dirichlet BCs)
    bc_values = jnp.zeros((bc_coords.shape[0], 2))  # 2 outputs (u, v)

    return bc_coords, bc_values

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