from functools import partial
import os
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, debug, device_get
from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt


class LevelSet(ForwardIVP):
    def __init__(self, config, p0, t_star, x_star, y_star, time_offset=None, mask=None):
        super().__init__(config)

        self.p0 = p0
        self.t_star = t_star
        self.x_star = x_star
        self.y_star = y_star
        self.time_offset = time_offset
        self.exact_mass = (jnp.pi * 0.15**2) - (0.25*0.05) # Area of circle
        self.mask = mask
        # self.bc_coords = bc_coords
        # self.bc_values = bc_values

        # Predictions over a grid
        self.p0_pred_fn = vmap(vmap(self.p_net, (None, None, None, 0)), (None, None, 0, None))
        self.p_bc_pred_fn = vmap(self.p_net, (None, 0, 0, 0))
        self.eik_pred_fn = vmap(self.eik_net, (None, 0, 0, 0))
        self.p_pred_fn = vmap(vmap(vmap(self.p_net, (None, None, None, 0)), (None, None, 0, None)), (None, 0, None, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0))

    def neural_net(self, params, t, x, y):
        # t = t / self.t_star[-1]
        inputs = jnp.stack([t, x, y])
        _, outputs = self.state.apply_fn(params, inputs)

        p = outputs[0]
        return p
    
    def p_net(self, params, t, x, y):
        p = self.neural_net(params, t, x, y)
        return p
    
    def f_vx(self, t, x, y):    # zalesak disk
        if self.config.nondim.nondimensionalize == True:
            t = t * self.config.nondim.T_star
            x = x * self.config.nondim.X_star
            y = y * self.config.nondim.Y_star

        omega = jnp.pi / 3.14
        vx = omega * (0.5 - y)

        if self.config.nondim.nondimensionalize == True:
            vx = vx / self.config.nondim.V_star
        return vx

    def f_vy(self, t, x, y):    # zalesak disk
        if self.config.nondim.nondimensionalize == True:
            t = t * self.config.nondim.T_star
            x = x * self.config.nondim.X_star
            y = y * self.config.nondim.Y_star

        omega = jnp.pi / 3.14
        vy = omega * (x - 0.5)

        if self.config.nondim.nondimensionalize == True:
            vy = vy / self.config.nondim.V_star
        return vy

    def r_net(self, params, t, x, y):
        # p = self.p_net(params, t, x, y)

        # Compute gradients of the scaled level set field
        p_t = grad(self.p_net, argnums=1)(params, t, x, y)
        p_x = grad(self.p_net, argnums=2)(params, t, x, y)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y)

        # Calculate vx,vy accounting for the reset in time to [0,t] on each new time window
        if self.time_offset == None:
            vx = self.f_vx(t,x,y)
            vy = self.f_vy(t,x,y)
            # debug.print("t: {t}",t=t)
            # debug.print("vx: {vx}",vx=vx)
        else:
            t_offset = self.time_offset + t
            vx = self.f_vx(t_offset,x,y)
            vy = self.f_vy(t_offset,x,y)
            # debug.print("t_offset: {t_offset}",t_offset=t_offset)
            # debug.print("vx: {vx}",vx=vx)

        # Level set equation residual
        if self.config.nondim.nondimensionalize == True:
            rp = p_t + (self.config.nondim.V_star * self.config.nondim.T_star / self.config.nondim.X_star) * (vx * p_x + vy * p_y)
        else:
            rp = p_t + vx * p_x + vy * p_y

        return rp

    def eik_net(self, params, t, x, y):
        # Compute gradients for the eikonal loss calculation (reinitialization condition)
        p_x = grad(self.p_net, argnums=2)(params, t, x, y)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y)

        eik = jnp.sqrt(p_x**2 + p_y**2) - 1.0 # magnitude

        # debug.print("eik: {eik}",eik=eik)
        return eik
    
    def mass_net(self, params, t, x, y):
        # Mass loss
        p_pred = self.p_pred_fn(params, t, x, y) # Shape: (4,4,4)

        # Calculate the number of negative points for each time step
        negative_counts = jnp.sum(p_pred < 0, axis=(1, 2))  # Shape: (4,)
        mean_counts = jnp.mean(negative_counts) # Shape: ()
        # debug.print("mean_counts: {mean_counts}",mean_counts=mean_counts)
        # Area within boundary (total "negative area") for each time step
        area_within_boundary = mean_counts / negative_counts.shape[0]**2  # Shape: ()
        # debug.print("area: {area_within_boundary}",area_within_boundary=area_within_boundary)
        # debug.print("neg_counts shape: {shape}",shape=negative_counts.shape[0])
        return area_within_boundary
    
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        rp_pred = self.r_pred_fn(params, t_sorted, batch[:, 1], batch[:, 2])

        rp_pred = rp_pred.reshape(self.num_chunks, -1)

        rp_l = jnp.mean(rp_pred**2, axis=1)

        rp_gamma = lax.stop_gradient(jnp.exp(-self.tol * (self.M @ rp_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([rp_gamma])
        gamma = gamma.min(0)

        return rp_l, gamma

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # IC loss
        p0_pred = self.p0_pred_fn(params, 0.0, self.x_star, self.y_star) # (21,21)
        p0_loss = jnp.mean((p0_pred - self.p0) ** 2)

        # BC loss
        # p_bc_pred = self.p_bc_pred_fn(params, self.bc_coords[:, 0], self.bc_coords[:, 1], self.bc_coords[:, 2])
        # p_bc_loss = jnp.mean((p_bc_pred - self.bc_values[:, 0]) ** 2)

        # Eikonal loss
        if 'eik_p' in self.config.weighting.init_weights:
            eik_pred = self.eik_pred_fn(params, batch[:, 0], batch[:, 1], batch[:, 2])
            eik_loss = jnp.mean((eik_pred) ** 2)

        # Mass Loss (difference between computed area and exact mass)
        if 'mass_p' in self.config.weighting.init_weights:
            mass_pred = self.mass_net(params, batch[::20, 0], batch[::20, 1], batch[::20, 2])
            mass_loss = jnp.mean((mass_pred - self.exact_mass) ** 2)
            # debug.print("mass_loss: {mass_loss}",mass_loss=mass_loss)

        # Residual loss
        if self.config.weighting.use_causal == True:
            # batch (64,5) Uniform (spatial-time batch, t/x/y)
            res_batch = jnp.array([batch[:, 0].sort(), batch[:, 1], batch[:, 2]]).T   # (64,3)

            rp_l, gamma = self.res_and_w(params, res_batch)
            rp_loss = jnp.mean(rp_l * gamma)

        else:
            rp_pred = self.r_pred_fn(
                params, batch[:, 0], batch[:, 1], batch[:, 2]
            )
            # Compute loss
            rp_loss = jnp.mean(rp_pred**2)

        loss_dict = {
            "ic_p": p0_loss,
            # "p_bc": p_bc_loss,
            "rp": rp_loss,
        }
        if 'eik_p' in self.config.weighting.init_weights:
            loss_dict['eik_p'] = eik_loss
        if 'mass_p' in self.config.weighting.init_weights:
            loss_dict['mass_p'] = mass_loss
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_diag_ntk(self, params, batch):
        # Compute NTK for initial conditions (IC)
        ic_p_ntk = vmap(vmap(ntk_fn, (None, None, None, None, 0)), (None, None, None, 0, None))(self.p_net, params, 0.0, self.x_star, self.y_star)

        # Compute NTK for boundary conditions (BC)
        # p_bc_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(self.p_net, params, self.bc_coords[:, 0], self.bc_coords[:, 1], self.bc_coords[:, 2])

        # Consider the effect of causal weights
        if self.config.weighting.use_causal:
            batch = jnp.array([batch[:, 0].sort(), batch[:, 1], batch[:, 2]]).T
            rp_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(self.r_net, params, batch[:, 0], batch[:, 1], batch[:, 2])

            rp_ntk = rp_ntk.reshape(self.num_chunks, -1)

            rp_ntk = jnp.mean(rp_ntk, axis=1)

            _, casual_weights = self.res_and_w(params, batch)
            rp_ntk = rp_ntk * casual_weights
        else:
            rp_ntk = vmap(ntk_fn, (None, None, 0, 0, 0))(self.r_net, params, batch[:, 0], batch[:, 1], batch[:, 2])

        # ntk_dict = {"ic_p": ic_p_ntk, "p_bc": p_bc_ntk, "rp": rp_ntk}
        ntk_dict = {"ic_p": ic_p_ntk, "rp": rp_ntk}
        return ntk_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, x, y, p_ref):
        p_pred = self.p_pred_fn(params, t, x, y)

        # Apply mask to compute within circular region only of unit square
        mask = jnp.repeat(self.mask[jnp.newaxis, :, :], len(t), axis=0)
        p_pred_flat = p_pred.ravel()  # Shape (len(t) * ny * nx,)
        p_ref_flat = p_ref.ravel()    # Shape (len(t) * ny * nx,)
        mask_flat = mask.ravel()      # Shape (len(t) * ny * nx,)

        # Use jnp.where to select valid points
        p_pred = jnp.where(mask_flat, p_pred_flat, 0.0)
        p_ref = jnp.where(mask_flat, p_ref_flat, 0.0)

        p_error = jnp.linalg.norm(p_pred - p_ref) / jnp.linalg.norm(p_ref)

        return p_error


class LevelSetEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, p_ref):
        # Compute the L2 errors for p
        p_error = self.model.compute_l2_error(params, self.model.t_star, self.model.x_star, self.model.y_star, p_ref)
        self.log_dict["l2_p_error"] = p_error

    def log_preds(self, params):
        # Predict p over the grid
        p_pred = self.model.p_pred_fn(params, self.model.t_star, self.model.x_star, self.model.y_star)

        # Log predictions for p
        fig_p = plt.figure(figsize=(6, 5))
        plt.imshow(p_pred.T, cmap="jet", extent=[self.model.x_star.min(), self.model.x_star.max(), self.model.y_star.min(), self.model.y_star.max()])
        plt.colorbar()
        plt.title("Predicted p")
        self.log_dict["p_pred"] = fig_p
        plt.close()

    def __call__(self, state, batch, p_ref):
        # Initialize the log dictionary
        self.log_dict = super().__call__(state, batch)

        # Log causal weight if causal weighting is enabled
        if self.config.weighting.use_causal:
            _, causal_weight = self.model.res_and_w(state.params, batch)
            self.log_dict["cas_weight"] = causal_weight.min()

        # Log errors for p if enabled in the config
        if self.config.logging.log_errors:
            self.log_errors(state.params, p_ref)

        # Log predictions for p if enabled in the config
        if self.config.logging.log_preds:
            self.log_preds(state.params)

        # Log nonlinearities for Pirate
        if self.config.logging.log_nonlinearities:
            layer_keys = [
                key
                for key in state.params["params"].keys()
                if key.endswith(
                    tuple(
                        [f"Bottleneck_{i}" for i in range(self.config.arch.num_layers)]
                    )
                )
            ]
            for i, key in enumerate(layer_keys):
                self.log_dict[f"alpha_{i}"] = state.params["params"][key]["alpha"]

        return self.log_dict
    

class LevelSetMulti(ForwardIVP):
    def __init__(self, config, p0, t_star, x_star, y_star, time_offset=None, mask=None):
        super().__init__(config)

        self.p0 = p0
        self.t_star = t_star
        self.x_star = x_star
        self.y_star = y_star
        self.time_offset = time_offset
        self.exact_mass = (jnp.pi * 0.15**2) - (0.25*0.05) # Area circle - Area slit
        self.mask = mask

        # Predictions over a grid
        self.p0_pred_fn = vmap(vmap(self.p_net, (None, None, None, 0)), (None, None, 0, None))
        self.eik_pred_fn = vmap(self.eik_net, (None, 0, 0, 0))
        self.p_pred_fn = vmap(vmap(vmap(self.p_net, (None, None, None, 0)), (None, None, 0, None)), (None, 0, None, None))

    def neural_net(self, params, t, x, y):
        # t = t / self.t_star[-1]
        inputs = jnp.stack([t, x, y])
        _, outputs = self.state.apply_fn(params, inputs)

        p = outputs[0]
        return p
    
    def p_net(self, params, t, x, y):
        p = self.neural_net(params, t, x, y)
        return p

    def eik_net(self, params, t, x, y):
        # Compute gradients for the eikonal loss calculation (reinitialization condition)
        p_x = grad(self.p_net, argnums=2)(params, t, x, y)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y)

        eik = jnp.sqrt(p_x**2 + p_y**2) - 1.0 # magnitude

        # debug.print("eik: {eik}",eik=eik)
        return eik
    
    def mass_net(self, params, t, x, y):
        # Mass loss
        p_pred = self.p_pred_fn(params, t, x, y) # Shape: (n,n,n)
        # debug.print("t: {t}",t=t)

        # Calculate the number of negative points for each time step
        negative_counts = jnp.sum(p_pred < 0, axis=(1, 2))  # Shape: (n,) Negative counts for each time step
        mean_counts = jnp.mean(negative_counts) # Shape: ()

        # Area within boundary (total "negative area") for each time step
        area_within_boundary = mean_counts / (negative_counts.shape[0]**2)  # Shape: ()

        return area_within_boundary
    
    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch):
        # Eikonal loss
        if 'eik_p' in self.config.weighting1.init_weights:
            eik_pred = self.eik_pred_fn(params, batch[:, 0], batch[:, 1], batch[:, 2])
            eik_loss = jnp.mean((eik_pred) ** 2)

        # Mass Loss (difference between computed area and exact mass)
        if 'mass_p' in self.config.weighting1.init_weights:
            batch_ratio = 1
            mass_pred = self.mass_net(params, batch[::batch_ratio, 0], batch[::batch_ratio, 1], batch[::batch_ratio, 2])
            mass_loss = jnp.mean((mass_pred - self.exact_mass) ** 2)
            # debug.print("mass_loss: {mass_loss}",mass_loss=mass_loss)

        loss_dict = {}
        if 'eik_p' in self.config.weighting1.init_weights:
            loss_dict['eik_p'] = eik_loss
        if 'mass_p' in self.config.weighting1.init_weights:
            loss_dict['mass_p'] = mass_loss
        return loss_dict


    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, x, y, p_ref):
        p_pred = self.p_pred_fn(params, t, x, y)
        
        # Apply mask to compute within circular region only of unit square
        mask = jnp.repeat(self.mask[jnp.newaxis, :, :], len(t), axis=0)
        p_pred_flat = p_pred.ravel()
        p_ref_flat = p_ref.ravel()
        mask_flat = mask.ravel()      # Shape (nt * ny * nx,)

        # Use jnp.where to select valid points
        p_pred = jnp.where(mask_flat, p_pred_flat, 0.0)
        p_ref = jnp.where(mask_flat, p_ref_flat, 0.0)

        p_error = jnp.linalg.norm(p_pred - p_ref) / jnp.linalg.norm(p_ref)

        return p_error