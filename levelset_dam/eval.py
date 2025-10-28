import os
import ml_collections
import wandb
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from jaxpi.utils import restore_checkpoint
from absl import logging

import models
from utils import get_dataset, get_bc_coords, plot_re_schedule, re_schedule_step, re_schedule_sigmoid, geometric_sdf_reinit

def pred_ref_err_gif(var_pred, var_star, t_star, x_star, y_star, workdir, config, var_name):
    Nt, Nx, Ny = var_star.shape
    dt = t_star[1] - t_star[0]
    X, Y = jnp.meshgrid(x_star, y_star, indexing="ij")

    # Compute global vmin and vmax for consistent color scale
    vmin = jnp.minimum(var_star.min(), var_pred.min())
    vmax = jnp.maximum(var_star.max(), var_pred.max())
    abs_error = jnp.abs(var_star - var_pred)
    vmax_err = abs_error.max()

    if var_name == 'p' or var_name == 'p_reinit':
        vmax = min(vmax, 2500)
        vmin = max(vmin, -1000)

    # Set up the figure and axes
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # First row (p: Ground truth, Prediction, Absolute Error)
    pcm1 = axes[0].pcolormesh(X, Y, var_star[0], cmap='jet', shading='gouraud', vmin=vmin, vmax=vmax)
    pcm2 = axes[1].pcolormesh(X, Y, var_pred[0], cmap='jet', shading='gouraud', vmin=vmin, vmax=vmax)
    pcm3 = axes[2].pcolormesh(X, Y, jnp.abs(var_star[0] - var_pred[0]), cmap='jet', shading='gouraud', vmin=0, vmax=vmax_err)
    fig.colorbar(pcm1, ax=axes[0])
    fig.colorbar(pcm2, ax=axes[1])
    fig.colorbar(pcm3, ax=axes[2])

    # Titles for variable
    axes[0].set_title(f'Ground Truth {var_name}')
    axes[1].set_title(f'Predicted {var_name}')
    axes[2].set_title(f'Absolute Error {var_name}')
    
    # Function to update the plots for animation
    def update(frame):
        # Update pcolormesh for u
        pcm1.set_array(var_star[frame].ravel())
        pcm2.set_array(var_pred[frame].ravel())
        pcm3.set_array(jnp.abs(var_star[frame] - var_pred[frame]).ravel())

        # Add interface contour plot and grid points
        if len(axes[0].collections) > 1:  # Ensure not to remove pcolormesh collections
            # The last collection is the contour we added, so remove it
            for col in axes[0].collections[1:]:
                col.remove()
            for col in axes[1].collections[1:]:
                col.remove()
        axes[0].contour(X, Y, var_star[frame], levels=[0.01], colors='black', linewidths=1)
        # axes[0].scatter(X, Y, c='black', s=0.01, marker='o')
        axes[1].contour(X, Y, var_pred[frame], levels=[0.01], colors='black', linewidths=1)
        # axes[1].scatter(X, Y, c='black', s=0.01, marker='o')

        # Update titles
        time = dt * frame
        axes[0].set_title(f'Ground Truth {var_name} at time {time:.2f} s')
        axes[1].set_title(f'Predicted {var_name} at time {time:.2f} s')
        axes[2].set_title(f'Absolute Error {var_name} at time {time:.2f} s')

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
    file_name = f'ls_dam_break_{var_name}.gif'
    ani.save(os.path.join(save_dir, file_name), writer=animation.PillowWriter(fps=10))
    
    # Clear the figure to prevent overlap of subsequent samples
    plt.clf()    



def pred_ref_err_plot_over_time(var_preds, var_stars, t_star, x_star, y_star, workdir, config):
    """
    Plot the per-time *relative* L2 error between prediction and reference for all fields present in the dicts.
    Overlays phi, u, v, p on the same axes"""
    # Drop first time step
    for k in var_preds.keys():
        var_preds[k] = var_preds[k][1:]
    for k in var_stars.keys():
        var_stars[k] = var_stars[k][1:]
    t_star = t_star[1:]
    nt = t_star.shape[0]

    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)

    def rel_l2_series(pred, ref, eps=1e-12):
        """Relative L2(t) = ||pred-ref||_2 / (||ref||_2 + eps), computed at each time step."""
        pred = np.asarray(pred)
        ref  = np.asarray(ref)

        # Flatten spatial dims per time step
        pred_2d = pred.reshape(nt, -1)
        ref_2d  = ref.reshape(nt, -1)

        # Mask NaNs/Infs if any (ignore them in the norm)
        mask = np.isfinite(pred_2d) & np.isfinite(ref_2d)
        # compute per time step
        num = np.sqrt(np.sum(((pred_2d - ref_2d) * mask)**2, axis=1))
        den = np.sqrt(np.sum((ref_2d * mask)**2, axis=1))
        return num / (den + eps)

    # Compute error series for available keys
    keys_order = ["phi", "u", "v", "p"]
    errors = {}
    for k in keys_order:
        if k in var_preds and k in var_stars:
            errors[k] = rel_l2_series(var_preds[k], var_stars[k])

    # Plot all fields on one figure
    plt.figure(figsize=(9, 5))
    for k, e in errors.items():
        label = "ϕ" if k == "phi" else k
        plt.plot(t_star, e, label=label)

    plt.xlabel("Time (s)")
    plt.ylabel("Relative $L_2$ discrepancy")
    plt.title("Prediction vs Reference: per-time relative $L_2$ discrepancy")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    file_name = "all_fields_l2_evolution.png"
    out_path = os.path.join(save_dir, file_name)
    plt.savefig(out_path, dpi=200)
    plt.close()


def experimental_results_plot(var_preds, var_stars, t_star, x_star, y_star, workdir, config):
    """Compare LS-PINN with XFEM with experimental results from Martin and Moyce (1952)"""

    def interp_zero(x0, f0, x1, f1):
        """Linear interpolation to find x where f(x)=0 between (x0,f0) and (x1,f1)."""
        if f0 == 0.0:
            return x0
        if f1 == 0.0:
            return x1
        # fraction from point 0 to crossing where f=0
        alpha = f0 / (f0 - f1)
        return x0 + alpha * (x1 - x0)

    def extract_front_x_series(phi, x, y, y_band_height=None, phi_liquid_negative=True):
        """
        Extract surge-front x(t): furthest phi=0 crossing in a near-floor band.
        phi: (nt, nx, ny), x:(nx,), y:(ny,)
        Returns x_front_dim (nt,)
        """
        nt, nx, ny = phi.shape
        ymin = y.min()
        # Choose a thin band near the floor (default: max of 1% domain height or 3 grid steps)
        if y_band_height is None:
            # try 1% of total height or 3*dy, whichever is larger
            dy = np.min(np.diff(np.sort(y)))
            y_band_height = max(0.01 * (y.max() - y.min()), 3.0 * dy)
        y_band_mask = (y >= ymin) & (y <= ymin + y_band_height)
        y_idx = np.where(y_band_mask)[0]
        if y_idx.size == 0:
            raise ValueError("No points found in the near-floor band; check y grid and y_band_height.")

        x_front = np.full(nt, np.nan, dtype=float)

        for it in range(nt):
            x_candidates = []
            # For each small-y line, find the LAST crossing from liquid to air along +x
            for j in y_idx:
                line_phi = phi[it, :, j]
                # Define liquid as phi<0 if level set is signed distance (adjust if your convention differs)
                # We'll search for sign changes from negative to positive along x
                sign = np.sign(line_phi)
                # indices i where sign[i] <= 0 and sign[i+1] > 0  (crossing upward through zero)
                for i in range(len(x)-1):
                    f0, f1 = line_phi[i], line_phi[i+1]
                    # crossing if signs differ and it's liquid to air (negative to positive) OR exact zeros
                    if (f0 <= 0.0 and f1 > 0.0) or (f0 == 0.0) or (f1 == 0.0) or (f0 < 0.0 and f1 == 0.0):
                        # We want the *furthest downstream* crossing, so we append and will take max
                        try:
                            xc = interp_zero(x[i], f0, x[i+1], f1)
                            x_candidates.append(xc)
                        except ZeroDivisionError:
                            continue
            if x_candidates:
                x_front[it] = max(x_candidates)
            # else leave NaN (e.g., at t=0 before front forms)
        return x_front
    
    def extract_top_height_series(phi, x, y, x_band_width=None):
        """
        Extract top-of-column height ℓ(t): topmost phi=0 near the wall x ≈ xmin
        phi: (nt, nx, ny), x:(nx,), y:(ny,)
        Returns ell_dim (nt,)
        """
        nt, nx, ny = phi.shape
        xmin = x.min()
        if x_band_width is None:
            dx = np.min(np.diff(np.sort(x)))
            x_band_width = max(0.05 * (x.max() - x.min()), 3.0 * dx)  # 5% domain or 3*dx
        x_band_mask = (x >= xmin) & (x <= xmin + x_band_width)
        x_idx = np.where(x_band_mask)[0]
        if x_idx.size == 0:
            raise ValueError("No points found in the near-wall band; check x grid and x_band_width.")

        ell = np.full(nt, np.nan, dtype=float)

        for it in range(nt):
            y_candidates = []
            # For each small-x column, find the HIGHEST crossing from liquid to air along +y
            for i in x_idx:
                col_phi = phi[it, i, :]
                # scan along y for sign changes from negative (liquid) to positive (air)
                for j in range(len(y)-1):
                    f0, f1 = col_phi[j], col_phi[j+1]
                    if (f0 <= 0.0 and f1 > 0.0) or (f0 == 0.0) or (f1 == 0.0) or (f0 < 0.0 and f1 == 0.0):
                        try:
                            yc = interp_zero(y[j], f0, y[j+1], f1)
                            y_candidates.append(yc)
                        except ZeroDivisionError:
                            continue
            if y_candidates:
                ell[it] = max(y_candidates)
        return ell
    
    # Martin and Moyce parameters
    a   = 0.146   # [m] column width 
    n2  = 2.0
    n = np.sqrt(n2)
    g   = 9.8     # [m/s^2]

    ts_keep = 60  # Keep only first 60 time steps (before impact)
    t_star = t_star[:ts_keep] # Keep only time steps before impact
    
    phi_xfem = var_stars['phi'][:ts_keep]
    phi_pinn = var_preds['phi'][:ts_keep]

    # Compute series for each method
    x_front_xfem = extract_front_x_series(phi_xfem, x_star, y_star)
    x_front_pinn = extract_front_x_series(phi_pinn, x_star, y_star)

    h_col_xfem = extract_top_height_series(phi_xfem, x_star, y_star)
    h_col_pinn = extract_top_height_series(phi_pinn, x_star, y_star)

    # Experimental data from Martin and Moyce (1952)
    exper_x = np.loadtxt(os.path.join(workdir, "data", "martin_moyce_table2_x.txt"), delimiter=",", dtype=float)
    exper_y = np.loadtxt(os.path.join(workdir, "data", "martin_moyce_table6_y_n22.txt"), delimiter=",", dtype=float)

    # Convert to dimensionless units
    X_front_xfem = x_front_xfem / a
    X_front_pinn = x_front_pinn / a
    T_front = n * t_star * np.sqrt(g / a)
    H_col_xfem = h_col_xfem / (a * n2)
    H_col_pinn = h_col_pinn / (a * n2)
    T_col = t_star * np.sqrt(g / a)

    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    os.makedirs(save_dir, exist_ok=True)

    # Plot: x(t) (surge front) and y(t) (top of column)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)

    # 1) x over time
    ax = axes[0]
    ax.plot(T_front, X_front_pinn, label="LS-PINN", lw=2)
    ax.plot(T_front, X_front_xfem, label="XFEM", lw=2)
    if exper_x is not None and len(exper_x) > 0:
        ax.scatter(exper_x[:,0], exper_x[:,1], s=25, marker='^', label="Martin & Moyce (1952)", c="g")
    ax.set_xlabel("Time t*sqrt(ng/a) [-]")
    ax.set_ylabel("Surge-front position x/a [-]")
    ax.set_title("Front along the floor [Dimensionless]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2) y over time (top of column)
    ax = axes[1]
    ax.plot(T_col, H_col_pinn, label="LS-PINN", lw=2)
    ax.plot(T_col, H_col_xfem, label="XFEM", lw=2)
    if exper_y is not None and len(exper_y) > 0:
        ax.scatter(exper_y[:,0], exper_y[:,1], s=25, marker='^', label="Martin & Moyce (1952)", c="g")
    ax.set_xlabel("Time t*sqrt(g/a) [-]")
    ax.set_ylabel("Top-of-column height h/(n^2 a) [-]")
    ax.set_title("Height of residual column [Dimensionless]")
    ax.legend()
    ax.grid(True, alpha=0.3)

    file_name = "experimental_compar.png"
    out_path = os.path.join(save_dir, file_name)
    plt.savefig(out_path, dpi=200)
    plt.close()


def evaluate_s2s(config: ml_collections.ConfigDict, workdir: str):
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

    # if config.training.num_time_windows > 1:
    #     phi_star = phi_star[:-1, :]   # Remove last time step
    # phi0 = phi_star[0, :, :]
    phi0 = phi0_star
    p0 = p0_star
    u0 = u0_star
    v0 = v0_star

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

    # Define bc coords
    bc_coords = get_bc_coords(dom, t, x_star, y_star)

    # Restore model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

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

    model = models.LevelSet(config, phi0, p0, u0, v0, t_star, x_star, y_star, bc_coords, re_values)

    phi_pred_list = []
    p_pred_list = []
    u_pred_list = []
    v_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        phi = phi_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        u = u_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        v = v_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute L2 error
        l2_phi_error, l2_p_error, l2_u_error, l2_v_error = model.compute_l2_error(params, t, x_star, y_star, phi, p, u, v)
        logging.info("Time window: {}, phi error: {:.3e}".format(idx + 1, l2_phi_error))
        logging.info("Time window: {}, p error: {:.3e}".format(idx + 1, l2_p_error))
        logging.info("Time window: {}, u error: {:.3e}".format(idx + 1, l2_u_error))
        logging.info("Time window: {}, v error: {:.3e}".format(idx + 1, l2_v_error))

        phi_pred = model.phi_pred_fn(params, t, model.x_star, model.y_star)
        p_pred = model.p_pred_fn(params, t, model.x_star, model.y_star)
        u_pred = model.u_pred_fn(params, t, model.x_star, model.y_star)
        v_pred = model.v_pred_fn(params, t, model.x_star, model.y_star)

        phi_pred_list.append(phi_pred)
        p_pred_list.append(p_pred)
        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)

    # Get the full prediction
    phi_pred = jnp.concatenate(phi_pred_list, axis=0)
    p_pred = jnp.concatenate(p_pred_list, axis=0)
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)

    phi_error = jnp.linalg.norm(phi_pred - phi_star) / jnp.linalg.norm(phi_star)
    p_error = jnp.linalg.norm(p_pred - p_star) / jnp.linalg.norm(p_star)
    u_error = jnp.linalg.norm(u_pred - u_star) / jnp.linalg.norm(u_star)
    v_error = jnp.linalg.norm(v_pred - v_star) / jnp.linalg.norm(v_star)

    logging.info("L2 error of the full prediction of phi: {:.3e}".format(phi_error))
    logging.info("L2 error of the full prediction of p: {:.3e}".format(p_error))
    logging.info("L2 error of the full prediction of u: {:.3e}".format(u_error))
    logging.info("L2 error of the full prediction of v: {:.3e}".format(v_error))

    phi_mae = jnp.sum(jnp.abs(phi_pred - phi_star)) / jnp.sum(jnp.abs(phi_star))
    p_mae   = jnp.sum(jnp.abs(p_pred - p_star))   / jnp.sum(jnp.abs(p_star))
    u_mae   = jnp.sum(jnp.abs(u_pred - u_star))   / jnp.sum(jnp.abs(u_star))
    v_mae   = jnp.sum(jnp.abs(v_pred - v_star))   / jnp.sum(jnp.abs(v_star))

    logging.info("MAE rel of the full prediction of phi: {:.3e}".format(phi_mae))
    logging.info("MAE rel of the full prediction of p: {:.3e}".format(p_mae))
    logging.info("MAE rel of the full prediction of u: {:.3e}".format(u_mae))
    logging.info("MAE rel of the full prediction of v: {:.3e}".format(v_mae))

    # Dimensionalize coordinates and flow field
    if config.nondim.nondimensionalize == True:
        t_star = t_star * config.nondim.T_star
        x_star = x_star * config.nondim.X_star
        y_star = y_star * config.nondim.Y_star
        phi_star = phi_star * config.nondim.PHI_star
        phi_pred = phi_pred * config.nondim.PHI_star
        p0_star = p0_star * config.nondim.P_star
        u0_star = u0_star * config.nondim.U_star
        v0_star = v0_star * config.nondim.V_star
        p_star = p_star * config.nondim.P_star
        p_pred = p_pred * config.nondim.P_star
        u_star = u_star * config.nondim.U_star
        u_pred = u_pred * config.nondim.U_star
        v_star = v_star * config.nondim.V_star
        v_pred = v_pred * config.nondim.V_star

    preds = {
        "phi": phi_pred,
        "u":   u_pred,
        "v":   v_pred,
        "p":   p_pred,
    }

    stars = {
        "phi": phi_star,
        "u":   u_star,
        "v":   v_star,
        "p":   p_star,
    }

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Compare with experimental results
    experimental_results_plot(preds, stars, t_star, x_star, y_star, workdir, config)
    # wandb.log({f"Dam break experimental comparison": wandb.Image(os.path.join(save_dir, "experimental_compar.png"))})

    # Plot L2 error over time for all fields
    pred_ref_err_plot_over_time(preds, stars, t_star, x_star, y_star, workdir, config)
    wandb.log({f"L2 over time": wandb.Image(os.path.join(save_dir, "all_fields_l2_evolution.png"))})

    # Save GIFs for phi, p, u, v
    pred_ref_err_gif(phi_pred, phi_star, t_star, x_star, y_star, workdir, config, 'phi')
    pred_ref_err_gif(p_pred, p_star, t_star, x_star, y_star, workdir, config, 'p')
    pred_ref_err_gif(u_pred, u_star, t_star, x_star, y_star, workdir, config, 'u')
    pred_ref_err_gif(v_pred, v_star, t_star, x_star, y_star, workdir, config, 'v')

    wandb.log({f"LS dam break phi": wandb.Video(os.path.join(save_dir, 'ls_dam_break_phi.gif'))})
    wandb.log({f"LS dam break p": wandb.Video(os.path.join(save_dir, 'ls_dam_break_p.gif'))})
    wandb.log({f"LS dam break u": wandb.Video(os.path.join(save_dir, 'ls_dam_break_u.gif'))})
    wandb.log({f"LS dam break v": wandb.Video(os.path.join(save_dir, 'ls_dam_break_v.gif'))})

    # Plot evolution of re
    if config.training.re_schedule != None:
        plot_re_schedule(config, workdir, config.training.re_schedule, config.nondim.Re)


def plot_sliced_results(config: ml_collections.ConfigDict, workdir: str):
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

    # if config.training.num_time_windows > 1:
    #     phi_star = phi_star[:-1, :]   # Remove last time step
    # phi0 = phi_star[0, :, :]
    phi0 = phi0_star
    p0 = p0_star
    u0 = u0_star
    v0 = v0_star

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

    # Define bc coords
    bc_coords = get_bc_coords(dom, t, x_star, y_star)

    # Restore model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

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

    model = models.LevelSet(config, phi0, p0, u0, v0, t_star, x_star, y_star, bc_coords, re_values)

    phi_pred_list = []
    p_pred_list = []
    u_pred_list = []
    v_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        phi = phi_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        u = u_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        v = v_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute L2 error
        l2_phi_error, l2_p_error, l2_u_error, l2_v_error = model.compute_l2_error(params, t, x_star, y_star, phi, p, u, v)
        logging.info("Time window: {}, phi error: {:.3e}".format(idx + 1, l2_phi_error))
        logging.info("Time window: {}, p error: {:.3e}".format(idx + 1, l2_p_error))
        logging.info("Time window: {}, u error: {:.3e}".format(idx + 1, l2_u_error))
        logging.info("Time window: {}, v error: {:.3e}".format(idx + 1, l2_v_error))

        phi_pred = model.phi_pred_fn(params, t, model.x_star, model.y_star)
        p_pred = model.p_pred_fn(params, t, model.x_star, model.y_star)
        u_pred = model.u_pred_fn(params, t, model.x_star, model.y_star)
        v_pred = model.v_pred_fn(params, t, model.x_star, model.y_star)

        phi_pred_list.append(phi_pred)
        p_pred_list.append(p_pred)
        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)

    # Get the full prediction
    phi_pred = jnp.concatenate(phi_pred_list, axis=0)
    p_pred = jnp.concatenate(p_pred_list, axis=0)
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)

    phi_error = jnp.linalg.norm(phi_pred - phi_star) / jnp.linalg.norm(phi_star)

    logging.info("L2 error of the full prediction of phi: {:.3e}".format(phi_error))

    # Dimensionalize coordinates and flow field
    if config.nondim.nondimensionalize == True:
        t_star = t_star * config.nondim.T_star
        x_star = x_star * config.nondim.X_star
        y_star = y_star * config.nondim.Y_star
        phi_star = phi_star * config.nondim.PHI_star
        p0_star = p0_star * config.nondim.P_star
        u0_star = u0_star * config.nondim.U_star
        v0_star = v0_star * config.nondim.V_star
        phi_pred = phi_pred * config.nondim.PHI_star
        p_pred = p_pred * config.nondim.P_star
        u_pred = u_pred * config.nondim.U_star
        v_pred = v_pred * config.nondim.V_star

    Nt, Nx, Ny = phi_ref.shape
    dt = t_star[1] - t_star[0]
    X, Y = jnp.meshgrid(x_star, y_star, indexing="ij")

    # Set time steps to plot
    ts = [0,10,20,30,40]

    # Compute global max error for uniform colorbar scaling
    max_abs_error = jnp.max(jnp.abs(phi_ref - phi_pred))
    min_phi = min(jnp.min(phi_pred),jnp.min(phi_ref))
    max_phi = max(jnp.max(phi_pred),jnp.max(phi_ref))

    # Set up the figure and axes
    fig, axes = plt.subplots(len(ts), 3, figsize=(18, 5*len(ts)))

    for idx in range(len(ts)):
        # Row (p: Ground truth, Prediction, Absolute Error)
        pcm1 = axes[idx, 0].pcolormesh(X, Y, phi_ref[ts[idx]], cmap='jet', shading='gouraud', vmin=min_phi, vmax=max_phi)
        pcm2 = axes[idx, 1].pcolormesh(X, Y, phi_pred[ts[idx]], cmap='jet', shading='gouraud', vmin=min_phi, vmax=max_phi)
        pcm3 = axes[idx, 2].pcolormesh(X, Y, jnp.abs(phi_ref[ts[idx]] - phi_pred[ts[idx]]), cmap='jet', shading='gouraud', vmin=0, vmax=max_abs_error)
        fig.colorbar(pcm1, ax=axes[idx, 0])
        fig.colorbar(pcm2, ax=axes[idx, 1])
        fig.colorbar(pcm3, ax=axes[idx, 2])

        axes[idx, 0].contour(X, Y, phi_ref[ts[idx]], levels=[0.01], colors='black', linewidths=1)
        axes[idx, 1].contour(X, Y, phi_pred[ts[idx]], levels=[0.01], colors='black', linewidths=1)

        # Titles for phi
        axes[idx, 0].set_title(f'Reference phi t = {round(ts[idx]*dt,2)} s')
        axes[idx, 1].set_title(f'Predicted phi t = {round(ts[idx]*dt,2)} s')
        axes[idx, 2].set_title(f'Absolute Error phi t = {round(ts[idx]*dt,2)} s')

        for idx2 in range(3):
            axes[idx, idx2].set_xlabel('x')
            axes[idx, idx2].set_ylabel('y')
    
    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save the png
    file_name = f'levelset_phi_ts.png'
    plt.savefig(os.path.join(save_dir, file_name))
    plt.clf

    # Set up the figure with 4 rows (phi, p, u, v) and columns for time steps
    fig, axes = plt.subplots(4, len(ts), figsize=(6*len(ts), 5*4))  # 4 rows, len(ts) columns

    # Create lists of variables and their predictions
    variables = [
        ('phi', phi_pred, min_phi, max_phi),
        ('p', p_pred, jnp.min(p_pred), jnp.max(p_pred)),
        ('u', u_pred, jnp.min(u_pred), jnp.max(u_pred)),
        ('v', v_pred, jnp.min(v_pred), jnp.max(v_pred))
    ]

    for row_idx, (var_name, var_pred, vmin, vmax) in enumerate(variables):
        for col_idx, t_idx in enumerate(ts):
            ax = axes[row_idx, col_idx]
            pcm = ax.pcolormesh(X, Y, var_pred[t_idx], 
                              cmap='jet', shading='gouraud',
                              vmin=vmin, vmax=vmax)
            fig.colorbar(pcm, ax=ax)
            
            # Add contour lines for phi
            if var_name == 'phi':
                ax.contour(X, Y, var_pred[t_idx], levels=[0.01], colors='black', linewidths=1)
            
            # Set labels and titles
            if row_idx == 0:  # Only show time in top row titles
                ax.set_title(f't = {round(t_idx*dt,2)} s')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            
            # Add variable name to first column
            if col_idx == 0:
                ax.text(-0.35, 0.5, var_name, 
                       rotation=90, va='center', ha='center',
                       transform=ax.transAxes, fontsize=12)
            
            # Add variable name to first column
            if col_idx == 0:
                ax.text(-0.25, 0.5, var_name, 
                       rotation=0, va='center', ha='center',
                       transform=ax.transAxes, fontsize=12,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()

    # Save the figure
    file_name = f'levelset_phi_ts_pred.png'
    plt.savefig(os.path.join(save_dir, file_name))
    plt.clf()

    wandb.log({f"LS Time Slices {len(ts)}": wandb.Image(os.path.join(save_dir, file_name))})


def plot_mass_loss(config: ml_collections.ConfigDict, workdir: str):
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

    # if config.training.num_time_windows > 1:
    #     phi_star = phi_star[:-1, :]   # Remove last time step
    # phi0 = phi_star[0, :, :]
    phi0 = phi0_star
    p0 = p0_star
    u0 = u0_star
    v0 = v0_star

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

    # Define bc coords
    bc_coords = get_bc_coords(dom, t, x_star, y_star)

    # Restore model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

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

    model = models.LevelSet(config, phi0, p0, u0, v0, t_star, x_star, y_star, bc_coords, re_values)

    phi_pred_list = []
    p_pred_list = []
    u_pred_list = []
    v_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        phi = phi_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        u = u_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        v = v_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute L2 error
        l2_phi_error, l2_p_error, l2_u_error, l2_v_error = model.compute_l2_error(params, t, x_star, y_star, phi, p, u, v)
        logging.info("Time window: {}, phi error: {:.3e}".format(idx + 1, l2_phi_error))
        logging.info("Time window: {}, p error: {:.3e}".format(idx + 1, l2_p_error))
        logging.info("Time window: {}, u error: {:.3e}".format(idx + 1, l2_u_error))
        logging.info("Time window: {}, v error: {:.3e}".format(idx + 1, l2_v_error))

        phi_pred = model.phi_pred_fn(params, t, model.x_star, model.y_star)
        p_pred = model.p_pred_fn(params, t, model.x_star, model.y_star)
        u_pred = model.u_pred_fn(params, t, model.x_star, model.y_star)
        v_pred = model.v_pred_fn(params, t, model.x_star, model.y_star)

        phi_pred_list.append(phi_pred)
        p_pred_list.append(p_pred)
        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)

    # Get the full prediction
    phi_pred = jnp.concatenate(phi_pred_list, axis=0)
    p_pred = jnp.concatenate(p_pred_list, axis=0)
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)

    phi_error = jnp.linalg.norm(phi_pred - phi_star) / jnp.linalg.norm(phi_star)
    p_error = jnp.linalg.norm(p_pred - p_star) / jnp.linalg.norm(p_star)
    u_error = jnp.linalg.norm(u_pred - u_star) / jnp.linalg.norm(u_star)
    v_error = jnp.linalg.norm(v_pred - v_star) / jnp.linalg.norm(v_star)

    logging.info("L2 error of the full prediction of phi: {:.3e}".format(phi_error))
    logging.info("L2 error of the full prediction of p: {:.3e}".format(p_error))
    logging.info("L2 error of the full prediction of u: {:.3e}".format(u_error))
    logging.info("L2 error of the full prediction of v: {:.3e}".format(v_error))

    # Dimensionalize coordinates and flow field
    if config.nondim.nondimensionalize == True:
        t_star = t_star * config.nondim.T_star
        x_star = x_star * config.nondim.X_star
        y_star = y_star * config.nondim.Y_star
        phi_star = phi_star * config.nondim.PHI_star
        phi_pred = phi_pred * config.nondim.PHI_star
        p0_star = p0_star * config.nondim.P_star
        u0_star = u0_star * config.nondim.U_star
        v0_star = v0_star * config.nondim.V_star
        p_star = p_star * config.nondim.P_star
        p_pred = p_pred * config.nondim.P_star
        u_star = u_star * config.nondim.U_star
        u_pred = u_pred * config.nondim.U_star
        v_star = v_star * config.nondim.V_star
        v_pred = v_pred * config.nondim.V_star

    # Calculate mass ref
    negative_counts_ref = jnp.sum(phi_ref < 0, axis=(1, 2))
    mean_counts_ref = jnp.mean(negative_counts_ref)

    # Calulate mass loss
    negative_counts_pred = jnp.sum(phi_pred < 0, axis=(1, 2))  # Shape: (4,)
    mean_counts_pred = jnp.mean(negative_counts_pred) # Shape: ()
    # debug.print("mean_counts: {mean_counts}",mean_counts=mean_counts)
    # Area within boundary (total "negative area") for each time step
    # area_within_boundary = mean_counts / negative_counts.shape[0]  # Shape: ()

    negative_counts_err = jnp.abs(negative_counts_ref - negative_counts_pred)
    absolute_error = negative_counts_err / negative_counts_ref
    absolute_pct_error = absolute_error * 100
    mape = jnp.mean(jnp.abs((negative_counts_ref - negative_counts_pred) / negative_counts_ref)) * 100
    
    print(f"absolute_pct_error: {absolute_pct_error}")
    print(f"MAPE: {mape}")
    
    # Focus y-axis around relevant data scale
    y_min = max(0, absolute_pct_error.min() - 1)
    y_max = absolute_pct_error.max() + 1

    # Create the plot
    plt.figure(figsize=(8, 5))
    plt.plot(t_ref, absolute_pct_error, label="PINN Absolute Percent Error", color='blue', linewidth=2)
    plt.plot(t_ref, fem_mass_ape, label="FEM Absolute Percent Error", color='red', linewidth=2)
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

    wandb.log({f"LS mass loss": wandb.Image(os.path.join(save_dir, file_name))})


def plot_heaviside_and_density(config: ml_collections.ConfigDict, workdir: str):
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

    # Initialize model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

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

    model = models.LevelSet(config, phi0, p0, u0, v0, t_star, x_star, y_star, bc_coords, re_values)

    # Time steps to evaluate
    ts = [0, 10, 20, 30, 40]
    Nt, Nx, Ny = len(t_star), len(x_star), len(y_star)
    X, Y = jnp.meshgrid(x_star, y_star, indexing="ij")

    # Prepare storage
    H_pred_list = []
    rho_pred_list = []
    phi_pred_list = []

    for idx in range(config.training.num_time_windows):
        ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", f"time_window_{idx + 1}")
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Predict phi
        phi_pred = model.phi_pred_fn(params, t, model.x_star, model.y_star)

        # Compute Heaviside and density
        H_pred = 0.5 * (1 + jnp.tanh(phi_pred / config.hs_epsilon))
        rho_pred = config.rho1 * (1 - H_pred) + config.rho2 * (H_pred)
        logging.info("rho_pred min: {:.3e}".format(rho_pred.min()))
        logging.info("rho_pred max: {:.3e}".format(rho_pred.max()))

        H_pred_list.append(H_pred)
        rho_pred_list.append(rho_pred)
        phi_pred_list.append(phi_pred)

    H_pred = jnp.concatenate(H_pred_list, axis=0)
    rho_pred = jnp.concatenate(rho_pred_list, axis=0)
    phi_pred = jnp.concatenate(phi_pred_list, axis=0)

    # Dimensionalize coordinates and flow field
    if config.nondim.nondimensionalize == True:
        t_star = t_star * config.nondim.T_star
        x_star = x_star * config.nondim.X_star
        y_star = y_star * config.nondim.Y_star
        phi_star = phi_star * config.nondim.PHI_star
        phi_pred = phi_pred * config.nondim.PHI_star
        p0_star = p0_star * config.nondim.P_star
        u0_star = u0_star * config.nondim.U_star
        v0_star = v0_star * config.nondim.V_star

    H_ref = 0.5 * (1 + jnp.tanh(phi_ref / config.hs_epsilon))
    rho_ref = config.rho1 * (1 - H_ref) + config.rho2 * (H_ref)

    logging.info("rho_pred min: {:.3e}".format(rho_pred.min()))
    logging.info("rho_pred max: {:.3e}".format(rho_pred.max()))

    # Plot: now 3 rows: H(phi), rho (PINN), rho (FEM)
    fields = [
        ('Heaviside (PINN)', H_pred),
        ('Density (PINN)', rho_pred),
        ('Phi (PINN)', phi_pred),
        ('Heaviside (Ref)', H_ref),
        ('Density (Ref)', rho_ref),
        ('Phi (Ref)', phi_ref),
    ]
    fig, axes = plt.subplots(len(fields), len(ts), figsize=(6 * len(ts), 5 * len(fields)))  # 3 rows


    for row, (label, field) in enumerate(fields):
        for col, t_idx in enumerate(ts):
            ax = axes[row, col]
            pcm = ax.pcolormesh(X, Y, field[t_idx], cmap='viridis', shading='gouraud')
            fig.colorbar(pcm, ax=ax)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            if row == 0:
                ax.set_title(f't = {round(t_star[t_idx], 3)} s')
            if col == 0:
                ax.text(-0.15, 0.5, label, transform=ax.transAxes,
                        rotation=90, va='center', ha='center', fontsize=12,
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    plt.tight_layout()

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save the animation as a GIF
    file_name = f'density.png'
    plt.savefig(os.path.join(save_dir, file_name))

    wandb.log({f"Heaviside density": wandb.Image(os.path.join(save_dir, file_name))})


def evaluate_s2s_reinit(config: ml_collections.ConfigDict, workdir: str):
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

    # if config.training.num_time_windows > 1:
    #     phi_star = phi_star[:-1, :]   # Remove last time step
    # phi0 = phi_star[0, :, :]
    phi0 = phi0_star
    p0 = p0_star
    u0 = u0_star
    v0 = v0_star

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

    # Define bc coords
    bc_coords = get_bc_coords(dom, t, x_star, y_star)

    # Restore model
    if config.use_pi_init:
        config.arch.pi_init = jnp.zeros((config.arch.hidden_dim, config.arch.out_dim))

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

    model = models.LevelSet(config, phi0, p0, u0, v0, t_star, x_star, y_star, bc_coords, re_values)

    phi_pred_list = []
    p_pred_list = []
    u_pred_list = []
    v_pred_list = []

    for idx in range(config.training.num_time_windows):
        # Get the reference solution for the current time window
        phi = phi_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        p = p_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        u = u_star[num_time_steps * idx : num_time_steps * (idx + 1), :]
        v = v_star[num_time_steps * idx : num_time_steps * (idx + 1), :]

        ckpt_path = os.path.join(workdir, "checkpoints", config.wandb.name, "ckpt", "time_window_{}".format(idx + 1))
        model.state = restore_checkpoint(model.state, ckpt_path)
        params = model.state.params

        # Compute L2 error
        l2_phi_error, l2_p_error, l2_u_error, l2_v_error = model.compute_l2_error(params, t, x_star, y_star, phi, p, u, v)
        logging.info("Time window: {}, phi error: {:.3e}".format(idx + 1, l2_phi_error))
        logging.info("Time window: {}, p error: {:.3e}".format(idx + 1, l2_p_error))
        logging.info("Time window: {}, u error: {:.3e}".format(idx + 1, l2_u_error))
        logging.info("Time window: {}, v error: {:.3e}".format(idx + 1, l2_v_error))

        phi_pred = model.phi_pred_fn(params, t, model.x_star, model.y_star)
        p_pred = model.p_pred_fn(params, t, model.x_star, model.y_star)
        u_pred = model.u_pred_fn(params, t, model.x_star, model.y_star)
        v_pred = model.v_pred_fn(params, t, model.x_star, model.y_star)

        phi_pred_list.append(phi_pred)
        p_pred_list.append(p_pred)
        u_pred_list.append(u_pred)
        v_pred_list.append(v_pred)

    # Get the full prediction
    phi_pred = jnp.concatenate(phi_pred_list, axis=0)
    p_pred = jnp.concatenate(p_pred_list, axis=0)
    u_pred = jnp.concatenate(u_pred_list, axis=0)
    v_pred = jnp.concatenate(v_pred_list, axis=0)

    # Keep only reinit time steps
    reinit_steps = np.arange(0, len(t_star), len(t_star)/config.training.num_time_windows).astype(int)
    reinit_steps[0] = 1
    reinit_steps = np.append(reinit_steps, len(t_star) - 1)  # Ensure last step is included

    t_star = t_star[reinit_steps]

    phi_pred = phi_pred[reinit_steps]
    p_pred = p_pred[reinit_steps]
    u_pred = u_pred[reinit_steps]
    v_pred = v_pred[reinit_steps]

    phi_star = phi_star[reinit_steps]
    p_star = p_star[reinit_steps]
    u_star = u_star[reinit_steps]
    v_star = v_star[reinit_steps]

    phi_error = jnp.linalg.norm(phi_pred - phi_star) / jnp.linalg.norm(phi_star)
    p_error = jnp.linalg.norm(p_pred - p_star) / jnp.linalg.norm(p_star)
    u_error = jnp.linalg.norm(u_pred - u_star) / jnp.linalg.norm(u_star)
    v_error = jnp.linalg.norm(v_pred - v_star) / jnp.linalg.norm(v_star)

    logging.info("L2 error of the full prediction of phi (only reinit steps): {:.3e}".format(phi_error))
    logging.info("L2 error of the full prediction of p (only reinit steps): {:.3e}".format(p_error))
    logging.info("L2 error of the full prediction of u (only reinit steps): {:.3e}".format(u_error))
    logging.info("L2 error of the full prediction of v (only reinit steps): {:.3e}".format(v_error))

    phi_mae = jnp.sum(jnp.abs(phi_pred - phi_star)) / jnp.sum(jnp.abs(phi_star))
    p_mae   = jnp.sum(jnp.abs(p_pred - p_star))   / jnp.sum(jnp.abs(p_star))
    u_mae   = jnp.sum(jnp.abs(u_pred - u_star))   / jnp.sum(jnp.abs(u_star))
    v_mae   = jnp.sum(jnp.abs(v_pred - v_star))   / jnp.sum(jnp.abs(v_star))

    logging.info("MAE rel of the full prediction of phi: {:.3e}".format(phi_mae))
    logging.info("MAE rel of the full prediction of p: {:.3e}".format(p_mae))
    logging.info("MAE rel of the full prediction of u: {:.3e}".format(u_mae))
    logging.info("MAE rel of the full prediction of v: {:.3e}".format(v_mae))


    # Dimensionalize coordinates and flow field
    if config.nondim.nondimensionalize == True:
        t_star = t_star * config.nondim.T_star
        x_star = x_star * config.nondim.X_star
        y_star = y_star * config.nondim.Y_star
        phi_star = phi_star * config.nondim.PHI_star
        phi_pred = phi_pred * config.nondim.PHI_star
        p0_star = p0_star * config.nondim.P_star
        u0_star = u0_star * config.nondim.U_star
        v0_star = v0_star * config.nondim.V_star
        p_star = p_star * config.nondim.P_star
        p_pred = p_pred * config.nondim.P_star
        u_star = u_star * config.nondim.U_star
        u_pred = u_pred * config.nondim.U_star
        v_star = v_star * config.nondim.V_star
        v_pred = v_pred * config.nondim.V_star

    # Save the figure
    save_dir = os.path.join(workdir, "figures", config.wandb.name)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    # Save GIFs for phi, p, u, v
    pred_ref_err_gif(phi_pred, phi_star, t_star, x_star, y_star, workdir, config, 'phi_reinit')
    pred_ref_err_gif(p_pred, p_star, t_star, x_star, y_star, workdir, config, 'p_reinit')
    pred_ref_err_gif(u_pred, u_star, t_star, x_star, y_star, workdir, config, 'u_reinit')
    pred_ref_err_gif(v_pred, v_star, t_star, x_star, y_star, workdir, config, 'v_reinit')

    wandb.log({f"LS dam break phi_reinit": wandb.Video(os.path.join(save_dir, 'ls_dam_break_phi_reinit.gif'))})
    wandb.log({f"LS dam break p_reinit": wandb.Video(os.path.join(save_dir, 'ls_dam_break_p_reinit.gif'))})
    wandb.log({f"LS dam break u_reinit": wandb.Video(os.path.join(save_dir, 'ls_dam_break_u_reinit.gif'))})
    wandb.log({f"LS dam break v_reinit": wandb.Video(os.path.join(save_dir, 'ls_dam_break_v_reinit.gif'))})