import os
import ml_collections
import wandb
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.animation as animation
from jaxpi.utils import restore_checkpoint
from absl import logging

import models
from utils import get_dataset, get_bc_coords_values


def evaluate_s2s(config: ml_collections.ConfigDict, workdir: str):
    p_ref, t_ref, x_ref, y_ref, p0, fem_mass_ape = get_dataset(os.path.join("data",config.dataset))

    if config.nondim.nondimensionalize == True:
        t_star = t_ref / config.nondim.T_star
        x_star = x_ref / config.nondim.X_star
        y_star = y_ref / config.nondim.Y_star
        p_star = p_ref / config.nondim.P_star
        p0     = p0    / config.nondim.P_star
    else:
        t_star = t_ref
        x_star = x_ref
        y_star = y_ref
        p_star = p_ref

    if config.training.num_time_windows > 1:
        p_star = p_star[:-1, :]   # Remove last time step

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Create a mask for points inside the circular region
    X, Y = jnp.meshgrid(x_star, y_star, indexing='ij')
    center = jnp.array([config.mask.center_x, config.mask.center_y])  # Center of the circular region
    radius = config.mask.radius  # Radius of the circular region
    distance = jnp.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = distance <= radius

    # Restore model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    model = models.LevelSet(config, p0, t, x_star, y_star, mask=mask)

    p_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        if config.multi == True:
            model = models.LevelSetMulti(config, p0, t, x_star, y_star, mask=mask)
            ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}_level_1".format(idx + 1))
        else:
            ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute L2 error
        l2_p_error = model.compute_l2_error(params, t, x_star, y_star, p)
        logging.info("Time window: {}, p error: {:.3e}".format(idx + 1, l2_p_error))

        p_pred = model.p_pred_fn(params, model.t_star, model.x_star, model.y_star)

        p_pred_list.append(p_pred)

    # Get the full prediction
    p_pred = jnp.concatenate(p_pred_list, axis=0)

    # Apply mask to compute within circular region only of unit square
    mask = jnp.repeat(mask[jnp.newaxis, :, :], len(t), axis=0)
    p_pred_flat = p_pred.ravel()  # Shape (len(t) * ny * nx,)
    p_star_flat = p_star.ravel()    # Shape (len(t) * ny * nx,)
    mask_flat = mask.ravel()      # Shape (len(t) * ny * nx,)
    p_pred_masked = jnp.where(mask_flat, p_pred_flat, 0.0)
    p_star_masked = jnp.where(mask_flat, p_star_flat, 0.0)

    p_error = jnp.linalg.norm(p_pred_masked - p_star_masked) / jnp.linalg.norm(p_star_masked)

    logging.info("L2 error of the full prediction of p: {:.3e}".format(p_error))

    # Dimensionalize coordinates and flow field
    if config.nondim.nondimensionalize == True:
        t_star = t_star * config.nondim.T_star
        x_star = x_star * config.nondim.X_star
        y_star = y_star * config.nondim.Y_star
        p_star = p_star * config.nondim.P_star
        p_pred = p_pred * config.nondim.P_star
        p0     = p0     * config.nondim.P_star

    Nt, Nx, Ny = p_star.shape
    dt = t_star[1] - t_star[0]
    X, Y = jnp.meshgrid(x_star, y_star, indexing="ij")

    # Set up the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    p_star = jnp.array(p_star)
    p_pred = jnp.where(mask, p_pred, 0.0)
    p_star = jnp.where(mask, p_star, 0.0)

    # First row (p: Ground truth, Prediction, Absolute Error)
    pcm1 = axes[0].pcolormesh(X, Y, p_star[0], cmap='jet', shading='gouraud')
    pcm2 = axes[1].pcolormesh(X, Y, p_pred[0], cmap='jet', shading='gouraud')
    pcm3 = axes[2].pcolormesh(X, Y, jnp.abs(p_star[0] - p_pred[0]), cmap='jet', shading='gouraud')
    fig.colorbar(pcm1, ax=axes[0])
    fig.colorbar(pcm2, ax=axes[1])
    fig.colorbar(pcm3, ax=axes[2])

    # Titles for p
    axes[0].set_title('Ground Truth p')
    axes[1].set_title('Predicted p')
    axes[2].set_title('Absolute Error p')
    
    # Function to update the plots for animation
    def update(frame):
        # Update pcolormesh for u
        pcm1.set_array(p_star[frame].ravel())
        pcm2.set_array(p_pred[frame].ravel())
        pcm3.set_array(jnp.abs(p_star[frame] - p_pred[frame]).ravel())

        # Add interface contour plot and grid points
        if len(axes[0].collections) > 1:  # Ensure not to remove pcolormesh collections
            # The last collection is the contour we added, so remove it
            for col in axes[0].collections[1:]:
                col.remove()
            for col in axes[1].collections[1:]:
                col.remove()
        axes[0].contour(X, Y, p_star[frame], levels=[0.01], colors='black', linewidths=1)
        axes[0].scatter(X, Y, c='black', s=0.01, marker='o')
        axes[1].contour(X, Y, p_pred[frame], levels=[0.01], colors='black', linewidths=1)
        axes[1].scatter(X, Y, c='black', s=0.01, marker='o')

        # Update titles
        time = dt * frame
        axes[0].set_title(f'Ground Truth p at time {time:.2f} s')
        axes[1].set_title(f'Predicted p at time {time:.2f} s')
        axes[2].set_title(f'Absolute Error p at time {time:.2f} s')

        # Update labels
        axes[0].set_xlabel('$x$')
        axes[0].set_ylabel('$y$')
        axes[1].set_xlabel('$x$')
        axes[1].set_ylabel('$y$')
        axes[2].set_xlabel('$x$')
        axes[2].set_ylabel('$y$')
        axes[0].set_aspect('equal','box')
        axes[1].set_aspect('equal','box')
        axes[2].set_aspect('equal','box')
        
        return pcm1, pcm2, pcm3

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=Nt, blit=True)

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save the animation as a GIF
    file_name = f'level_set_zalesak.gif'
    ani.save(os.path.join(save_dir, file_name), writer=animation.PillowWriter(fps=10))
    
    # Clear the figure to prevent overlap of subsequent samples
    plt.clf()

    wandb.log({f"Level set fields": wandb.Video(os.path.join(save_dir, file_name))})


def plot_sliced_results(config: ml_collections.ConfigDict, workdir: str):
    p_ref, t_ref, x_ref, y_ref, p0, fem_mass_ape = get_dataset(os.path.join("data",config.dataset))
    if config.nondim.nondimensionalize == True:
        t_star = t_ref / config.nondim.T_star
        x_star = x_ref / config.nondim.X_star
        y_star = y_ref / config.nondim.Y_star
        p_star = p_ref / config.nondim.P_star
        p0     = p0    / config.nondim.P_star
    else:
        t_star = t_ref
        x_star = x_ref
        y_star = y_ref
        p_star = p_ref

    if config.training.num_time_windows > 1:
        p_star = p_star[:-1, :]   # Remove last time step

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Create a mask for points inside the circular region
    X, Y = jnp.meshgrid(x_star, y_star, indexing='ij')
    center = jnp.array([config.mask.center_x, config.mask.center_y])  # Center of the circular region
    radius = config.mask.radius  # Radius of the circular region
    distance = jnp.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = distance <= radius

    # Restore model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    model = models.LevelSet(config, p0, t, x_star, y_star, mask=mask)

    p_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        if config.multi == True:
            ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}_level_1".format(idx + 1))
        else:
            ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute L2 error
        l2_p_error = model.compute_l2_error(params, t, x_star, y_star, p)
        logging.info("Time window: {}, p error: {:.3e}".format(idx + 1, l2_p_error))

        p_pred = model.p_pred_fn(params, model.t_star, model.x_star, model.y_star)

        p_pred_list.append(p_pred)

    # Get the full prediction
    p_pred = jnp.concatenate(p_pred_list, axis=0)

    # Apply mask to compute within circular region only of unit square
    mask = jnp.repeat(mask[jnp.newaxis, :, :], len(t), axis=0)
    p_pred_flat = p_pred.ravel()  # Shape (len(t) * ny * nx,)
    p_star_flat = p_star.ravel()    # Shape (len(t) * ny * nx,)
    mask_flat = mask.ravel()      # Shape (len(t) * ny * nx,)
    p_pred_masked = jnp.where(mask_flat, p_pred_flat, 0.0)
    p_star_masked = jnp.where(mask_flat, p_star_flat, 0.0)

    p_error = jnp.linalg.norm(p_pred_masked - p_star_masked) / jnp.linalg.norm(p_star_masked)

    logging.info("L2 error of the full prediction of p: {:.3e}".format(p_error))

    # Dimensionalize coordinates and flow field
    if config.nondim.nondimensionalize == True:
        t_star = t_star * config.nondim.T_star
        x_star = x_star * config.nondim.X_star
        y_star = y_star * config.nondim.Y_star
        p_star = p_star * config.nondim.P_star
        p_pred = p_pred * config.nondim.P_star
        p0     = p0     * config.nondim.P_star

    Nt, Nx, Ny = p_star.shape
    dt = t_star[1] - t_star[0]
    X, Y = jnp.meshgrid(x_star, y_star, indexing="ij")

    # Set time steps to plot
    ts = [10,20,30,40]
    # ts = [0,25,50,75,100]

    # Compute global max error for uniform colorbar scaling
    max_abs_error = jnp.max(jnp.abs(p_star - p_pred))
    min_p = min(jnp.min(p_pred),jnp.min(p_star))
    max_p = max(jnp.max(p_pred),jnp.max(p_star))

    # Plot the reference, prediction and error colormaps
    # Set up the figure and axes
    fig, axes = plt.subplots(len(ts), 3, figsize=(18, 5*len(ts)))

    for idx in range(len(ts)):
        # Row (p: Ground truth, Prediction, Absolute Error)
        pcm1 = axes[idx, 0].pcolormesh(X, Y, p_star[ts[idx]], cmap='jet', shading='gouraud', vmin=min_p, vmax=max_p)
        pcm2 = axes[idx, 1].pcolormesh(X, Y, p_pred[ts[idx]], cmap='jet', shading='gouraud', vmin=min_p, vmax=max_p)
        pcm3 = axes[idx, 2].pcolormesh(X, Y, jnp.abs(p_star[ts[idx]] - p_pred[ts[idx]]), cmap='jet', shading='gouraud', vmin=0, vmax=max_abs_error)
        fig.colorbar(pcm1, ax=axes[idx, 0])
        fig.colorbar(pcm2, ax=axes[idx, 1])
        fig.colorbar(pcm3, ax=axes[idx, 2])

        # Add interface contour plot and grid points
        if len(axes[idx, 0].collections) > 1:  # Ensure not to remove pcolormesh collections
            # The last collection is the contour we added, so remove it
            for col in axes[idx, 0].collections[1:]:
                col.remove()
            for col in axes[idx, 1].collections[1:]:
                col.remove()
        axes[idx, 0].contour(X, Y, p_star[ts[idx]], levels=[0.01], colors='black', linewidths=1)
        axes[idx, 1].contour(X, Y, p_pred[ts[idx]], levels=[0.01], colors='black', linewidths=1)

        # Titles for p
        axes[idx, 0].set_title(f'Reference p t = {round(ts[idx]*dt,2)} s')
        axes[idx, 1].set_title(f'Predicted p t = {round(ts[idx]*dt,2)} s')
        axes[idx, 2].set_title(f'Absolute Error p t = {round(ts[idx]*dt,2)} s')

        for idx2 in range(3):
            axes[idx, idx2].set_xlabel('x')
            axes[idx, idx2].set_ylabel('y')
    
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save png
    file_name = f'levelset_p_ts.png'
    plt.savefig(os.path.join(save_dir, file_name))
    plt.clf

    # Plot the predictions only in a colormap
    # Set up the figure and axes
    fig, axes = plt.subplots(1, 1, figsize=(6, 6))

    for idx in range(len(ts)):
        ref_contour = axes.contour(X, Y, p_star[ts[idx]], levels=[0.01], colors='black', linewidths=1, linestyles='-')
        pred_contour = axes.contour(X, Y, p_pred[ts[idx]], levels=[0.01], colors='red', linewidths=2, linestyles=':')

    # Titles and labels
    axes.set_title(r'Zalesak disk Level Set Interfaces ($T=2\pi$)')
    axes.set_xlabel('x')
    axes.set_ylabel('y')

    # Add a legend
    reference_line = mlines.Line2D([], [], color='black', linestyle='-', linewidth=1, label='Reference Interface')
    predicted_line = mlines.Line2D([], [], color='red', linestyle=':', linewidth=2, label='Predicted Interface')
    axes.legend(handles=[reference_line, predicted_line], loc='upper right')

    # Add the text field at the bottom middle
    axes.text(0.05, 0.5, r'$\frac{T}{4}$', transform=axes.transAxes, fontsize=20, ha='center', va='top', color='Blue')
    axes.text(0.5, 0.08, r'$\frac{T}{2}$', transform=axes.transAxes, fontsize=20, ha='center', va='top', color='Blue')
    axes.text(0.95, 0.5, r'$\frac{3T}{4}$', transform=axes.transAxes, fontsize=20, ha='center', va='top', color='Blue')
    axes.text(0.5, 0.975, r'$T$', transform=axes.transAxes, fontsize=20, ha='center', va='top', color='Blue')

    plt.tight_layout()

    # Save png
    file_name = f'levelset_contour.png'
    plt.savefig(os.path.join(save_dir, file_name))

    wandb.log({f"Level set interfaces": wandb.Image(os.path.join(save_dir, file_name))})


def plot_mass_loss(config: ml_collections.ConfigDict, workdir: str):
    p_ref, t_ref, x_ref, y_ref, p0, fem_mass_ape = get_dataset(os.path.join("data",config.dataset))
    if config.nondim.nondimensionalize == True:
        t_star = t_ref / config.nondim.T_star
        x_star = x_ref / config.nondim.X_star
        y_star = y_ref / config.nondim.Y_star
        p_star = p_ref / config.nondim.P_star
        p0     = p0    / config.nondim.P_star
    else:
        t_star = t_ref
        x_star = x_ref
        y_star = y_ref
        p_star = p_ref

    if config.training.num_time_windows > 1:
        p_star = p_star[:-1, :]   # Remove last time step
        t_star = t_star[:-1, :]   # Remove last time step

    num_time_steps = len(t_star) // config.training.num_time_windows
    t = t_star[:num_time_steps]

    # Create a mask for points inside the circular region
    X, Y = jnp.meshgrid(x_star, y_star, indexing='ij')
    center = jnp.array([config.mask.center_x, config.mask.center_y])  # Center of the circular region
    radius = config.mask.radius  # Radius of the circular region
    distance = jnp.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask = distance <= radius

    # Restore model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

    model = models.LevelSet(config, p0, t, x_star, y_star, mask=mask)

    p_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        if config.multi == True:
            ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}_level_1".format(idx + 1))
        else:
            ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute L2 error
        l2_p_error = model.compute_l2_error(params, t, x_star, y_star, p)
        logging.info("Time window: {}, p error: {:.3e}".format(idx + 1, l2_p_error))

        p_pred = model.p_pred_fn(params, model.t_star, model.x_star, model.y_star)

        p_pred_list.append(p_pred)

    # Get the full prediction
    p_pred = jnp.concatenate(p_pred_list, axis=0)

    # Apply mask to compute within circular region only of unit square
    mask = jnp.repeat(mask[jnp.newaxis, :, :], len(t), axis=0)
    p_pred_flat = p_pred.ravel()  # Shape (len(t) * ny * nx,)
    p_star_flat = p_star.ravel()    # Shape (len(t) * ny * nx,)
    mask_flat = mask.ravel()      # Shape (len(t) * ny * nx,)
    p_pred_masked = jnp.where(mask_flat, p_pred_flat, 0.0)
    p_star_masked = jnp.where(mask_flat, p_star_flat, 0.0)

    p_error = jnp.linalg.norm(p_pred_masked - p_star_masked) / jnp.linalg.norm(p_star_masked)

    logging.info("L2 error of the full prediction of p: {:.3e}".format(p_error))

    # Dimensionalize coordinates and flow field
    if config.nondim == True:
        t_star = t_star * config.nondim.T_star
        x_star = x_star * config.nondim.X_star
        y_star = y_star * config.nondim.Y_star
        p_star = p_star * config.nondim.P_star
        p_pred = p_pred * config.nondim.P_star
        p0     = p0     * config.nondim.P_star

    # Calculate mass ref
    negative_counts_ref = jnp.sum(p_star < 0, axis=(1, 2))
    mean_counts_ref = jnp.mean(negative_counts_ref)

    # Calulate mass loss
    negative_counts_pred = jnp.sum(p_pred < 0, axis=(1, 2))  # Shape: (4,)
    mean_counts_pred = jnp.mean(negative_counts_pred) # Shape: ()
    # debug.print("mean_counts: {mean_counts}",mean_counts=mean_counts)
    # Area within boundary (total "negative area") for each time step
    # area_within_boundary = mean_counts / negative_counts.shape[0]  # Shape: ()

    negative_counts_err = jnp.abs(negative_counts_ref - negative_counts_pred)
    absolute_error = negative_counts_err / negative_counts_ref
    absolute_pct_error = absolute_error * 100
    mape = jnp.mean(jnp.abs((negative_counts_ref - negative_counts_pred) / negative_counts_ref)) * 100
    
    print(f"negative_counts_ref: {negative_counts_ref}")
    print(f"negative_counts_pred: {negative_counts_pred}")
    print(f"absolute_pct_error: {absolute_pct_error}")
    print(f"MAPE: {mape}")
    
    # Focus y-axis around relevant data scale
    y_min = max(0, absolute_pct_error.min() - 1)
    y_max = absolute_pct_error.max() + 1

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(t_star, absolute_pct_error, label="PINN Absolute Percent Error", color='blue', linewidth=2)
    plt.plot(t_star, fem_mass_ape, label="FEM Absolute Percent Error", color='red', linewidth=2)
    plt.xlabel("Time (s)", fontsize=12)
    plt.ylabel("Absolute Percent Error (%)", fontsize=12)
    plt.title("Level-set percent mass loss over time", fontsize=14)
    plt.ylim(y_min, y_max)
    plt.grid(alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save the animation as a GIF
    file_name = f'levelset_mass.png'
    plt.savefig(os.path.join(save_dir, file_name))

    wandb.log({f"Level set mass loss": wandb.Image(os.path.join(save_dir, file_name))})