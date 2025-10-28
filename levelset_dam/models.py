from functools import partial
import os
import jax.numpy as jnp
from jax import lax, jit, grad, vmap, jacrev, debug

from jaxpi.models import ForwardIVP
from jaxpi.evaluator import BaseEvaluator
from jaxpi.utils import ntk_fn, flatten_pytree

from matplotlib import pyplot as plt

from utils import re_schedule_step, re_schedule_sigmoid


class LevelSet(ForwardIVP):
    def __init__(self, config, phi0, p0, u0, v0, t_star, x_star, y_star, bc_coords, re_values, time_offset=None):
        super().__init__(config)

        self.phi0 = phi0
        self.p0 = p0
        self.u0 = u0
        self.v0 = v0
        self.t_star = t_star
        self.x_star = x_star
        self.y_star = y_star
        self.bc_coords = bc_coords
        self.time_offset = time_offset
        self.exact_mass = 0.146 * 0.292 # Area of dam
        self.rho1 = config.rho1 # Density (kg/m3) | Water
        self.rho2 = config.rho2 # Density (kg/m3) | Air
        self.mu1 = config.mu1   # Dynamic viscosity (Pa.s) | Water
        self.mu2 = config.mu2   # Dynamic viscosity (Pa.s) | Air
        # self.g = config.g       # Gravity (m/s2)
        self.re_values = re_values    # Gravity (m/s2) values according to g_scheduler
        self.Re = config.nondim.Re
        self.rho_ratio = config.nondim.rho_ratio
        self.mu_ratio = config.nondim.mu_ratio

        # Predictions over a grid
        self.phi0_pred_fn = vmap(vmap(self.phi_net, (None, None, None, 0)), (None, None, 0, None))
        self.p0_pred_fn = vmap(vmap(self.p_net, (None, None, None, 0)), (None, None, 0, None))
        self.u0_pred_fn = vmap(vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None))
        self.v0_pred_fn = vmap(vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None))
        self.phi_bc_pred_fn = vmap(self.phi_net, (None, 0, 0, 0))
        self.u_bc_pred_fn = vmap(self.u_net, (None, 0, 0, 0))
        self.v_bc_pred_fn = vmap(self.v_net, (None, 0, 0, 0))
        self.p_bc_pred_fn = vmap(self.p_net, (None, 0, 0, 0))
        self.eik_pred_fn = vmap(self.eik_net, (None, 0, 0, 0))
        self.phi_pred_fn = vmap(vmap(vmap(self.phi_net, (None, None, None, 0)), (None, None, 0, None)), (None, 0, None, None))
        self.p_pred_fn = vmap(vmap(vmap(self.p_net, (None, None, None, 0)), (None, None, 0, None)), (None, 0, None, None))
        self.u_pred_fn = vmap(vmap(vmap(self.u_net, (None, None, None, 0)), (None, None, 0, None)), (None, 0, None, None))
        self.v_pred_fn = vmap(vmap(vmap(self.v_net, (None, None, None, 0)), (None, None, 0, None)), (None, 0, None, None))
        self.r_pred_fn = vmap(self.r_net, (None, 0, 0, 0, None))

    def neural_net(self, params, t, x, y):
        # t = t / self.t_star[-1]
        inputs = jnp.stack([t, x, y])
        _, outputs = self.state.apply_fn(params, inputs)

        phi = outputs[0]
        p = outputs[1]
        u = outputs[2]
        v = outputs[3]

        # Gradient clipping
        u = lax.clamp(-self.config.training.grad_clip, u, self.config.training.grad_clip)
        v = lax.clamp(-self.config.training.grad_clip, v, self.config.training.grad_clip)
        p = lax.clamp(-self.config.training.grad_clip, p, self.config.training.grad_clip)
        phi = lax.clamp(-self.config.training.grad_clip, phi, self.config.training.grad_clip)
        
        return phi, p, u, v
    
    # Level set phi net
    def phi_net(self, params, t, x, y):
        phi, _, _, _ = self.neural_net(params, t, x, y)
        return phi
    
    # Pressure net
    def p_net(self, params, t, x, y):
        _, p, _, _ = self.neural_net(params, t, x, y)
        return p
    
    # Velocity net
    def u_net(self, params, t, x, y):
        _, _, u, _ = self.neural_net(params, t, x, y)
        return u
    
    # Velocity net
    def v_net(self, params, t, x, y):
        _, _, _, v = self.neural_net(params, t, x, y)
        return v
    
    def smoothed_heaviside(self, phi, epsilon=0.01):
        """Smoothed Heaviside function for density/viscosity transitions"""
        return 0.5 * (1 + jnp.tanh(phi / epsilon))

    def r_net(self, params, t, x, y, step):
        phi, p, u, v = self.neural_net(params, t, x, y)

        # Compute derivatives using automatic differentiation
        phi_t = grad(self.phi_net, argnums=1)(params, t, x, y)
        phi_x = grad(self.phi_net, argnums=2)(params, t, x, y)
        phi_y = grad(self.phi_net, argnums=3)(params, t, x, y)

        u_t = grad(self.u_net, argnums=1)(params, t, x, y)
        u_x = grad(self.u_net, argnums=2)(params, t, x, y)
        u_y = grad(self.u_net, argnums=3)(params, t, x, y)
        u_xx = grad(grad(self.u_net, argnums=2), argnums=2)(params, t, x, y)
        u_yy = grad(grad(self.u_net, argnums=3), argnums=3)(params, t, x, y)
        u_yx = grad(grad(self.u_net, argnums=3), argnums=2)(params, t, x, y)

        v_t = grad(self.v_net, argnums=1)(params, t, x, y)
        v_x = grad(self.v_net, argnums=2)(params, t, x, y)
        v_y = grad(self.v_net, argnums=3)(params, t, x, y)
        v_xx = grad(grad(self.v_net, argnums=2), argnums=2)(params, t, x, y)
        v_yy = grad(grad(self.v_net, argnums=3), argnums=3)(params, t, x, y)
        v_xy = grad(grad(self.v_net, argnums=2), argnums=3)(params, t, x, y)

        p_x = grad(self.p_net, argnums=2)(params, t, x, y)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y)

        # Gradually increase g with a scheduler
        if self.config.training.re_schedule == "step":
            re = self.re_values[step]
        elif self.config.training.re_schedule == "sigmoid":
            re = self.re_values[step]
        else:
            re = self.Re
        # debug.print("re: {re}", re=re)

        # # Claude version (dimensional)
        # # Material properties via smoothed Heaviside
        # eps = self.config.hs_epsilon
        # H = 0.5 * (1 + jnp.tanh(phi / eps))
        # dH_dphi = 0.5 * (1 - jnp.tanh(phi / eps) ** 2) / eps

        # rho = self.rho1 * (1 - H) + self.rho2 * (H)
        # mu = self.mu1 * (1 - H) + self.mu2 * (H)
        # mu_x = (self.mu2 - self.mu1) * dH_dphi * phi_x
        # mu_y = (self.mu2 - self.mu1) * dH_dphi * phi_y

        # # Residuals
        # r_phi = phi_t + u * phi_x + v * phi_y

        # r_cont = u_x + v_y

        # # Complete viscous stress terms
        # visc_x = (mu_x * u_x + mu * u_xx) + (mu_y * u_y + mu * u_yy) + (mu_x * v_y + mu * v_xy)
        # visc_y = (mu_x * v_x + mu * v_xx) + (mu_y * v_y + mu * v_yy) + (mu_y * u_x + mu * u_yx)
        
        # # x-momentum (non-conservative form)
        # r_mom_x = (u_t + u * u_x + v * u_y + 
        #         (1/rho) * p_x - 
        #         (1/rho) * visc_x)
        
        # # y-momentum (non-conservative form)  
        # r_mom_y = (v_t + u * v_x + v * v_y + 
        #         (1/rho) * p_y - 
        #         (1/rho) * visc_y - 
        #         g)  # gravity

        # Claude version (dimensionless)
        # Reynolds number (should be defined in your config)
        # Re = self.config.Re  # Re = rho_ref * sqrt(gL) * L / mu_ref

        # Dimensionless material properties via smoothed Heaviside
        eps = self.config.hs_epsilon
        H = 0.5 * (1 + jnp.tanh(phi / eps))
        dH_dphi = 0.5 * (1 - jnp.tanh(phi / eps) ** 2) / eps

        # Dimensionless density and viscosity ratios
        # rho_ratio = self.config.rho_ratio  # rho_air/rho_water ≈ 0.001
        # mu_ratio = self.config.mu_ratio    # mu_air/mu_water ≈ 0.02
        
        # Dimensionless material properties
        rho = (1 - H) + self.rho_ratio * H  # water: 1, air: rho_ratio
        mu = (1 - H) + self.mu_ratio * H    # water: 1, air: mu_ratio
        # debug.print("phi: {phi}",phi=phi)
        # debug.print("rho: {rho}",rho=rho)
        
        # Gradients of dimensionless viscosity
        mu_x = (self.mu_ratio - 1) * dH_dphi * phi_x
        mu_y = (self.mu_ratio - 1) * dH_dphi * phi_y

        # Dimensionless gravity components
        g_x = 0.0  # No horizontal gravity
        g_y = -1.0  # Gravity acts in negative y-direction

        # DIMENSIONLESS RESIDUALS
        
        # Level set residual
        r_phi = (phi_t + 
                u * phi_x + 
                v * phi_y)

        # Continuity residual
        r_cont = u_x + v_y

        # Viscous stress terms for x-momentum
        # visc_x = ((mu_x * u_x + mu * u_xx) + 
        #             (mu_y * u_y + mu * u_yy) + 
        #             (mu_x * v_y + mu * v_xy))
        visc_x = ((mu * u_xx) + (mu * u_yy))
        
        # Viscous stress terms for y-momentum  
        # visc_y = ((mu_x * v_x + mu * v_xx) + 
        #             (mu_y * v_y + mu * v_yy) + 
        #             (mu_y * u_x + mu * u_yx))
        visc_y = ((mu * v_xx) + (mu * v_yy))

        # x-momentum residual (dimensionless)
        r_mom_x = (u_t + 
                u * u_x + 
                v * u_y + 
                (1/rho) * p_x - 
                (1/re) * (1/rho) * visc_x - 
                g_x)
        
        # y-momentum residual (dimensionless)
        r_mom_y = (v_t + 
                u * v_x + 
                v * v_y + 
                (1/rho) * p_y - 
                (1/re) * (1/rho) * visc_y - 
                g_y)

        return r_phi, r_mom_x, r_mom_y, r_cont
    
    def r_phi_net(self, params, t, x, y):
        r_phi, _, _, _ = self.r_net(params, t, x, y)
        return r_phi
    
    def r_mom_x_net(self, params, t, x, y):
        _, r_mom_x, _, _ = self.r_net(params, t, x, y)
        return r_mom_x
    
    def r_mom_y_net(self, params, t, x, y):
        _, _, r_mom_y, _ = self.r_net(params, t, x, y)
        return r_mom_y
    
    def r_cont_net(self, params, t, x, y):
        _, _, _, r_cont = self.r_net(params, t, x, y)
        return r_cont

    def eik_net(self, params, t, x, y):
        # Compute gradients for the eikonal loss calculation (reinitialization condition)
        p_x = grad(self.p_net, argnums=2)(params, t, x, y)
        p_y = grad(self.p_net, argnums=3)(params, t, x, y)

        eik = jnp.sqrt(p_x**2 + p_y**2) - 1.0 # magnitude

        # debug.print("eik: {eik}",eik=eik)
        return eik
    
    def mass_net(self, params, t, x, y):
        # Mass loss
        p_pred = self.phi_pred_fn(params, t, x, y) # Shape: (4,4,4)

        # Calculate the number of negative points for each time step
        negative_counts = jnp.sum(p_pred < 0, axis=(1, 2))  # Shape: (4,)
        mean_counts = jnp.mean(negative_counts) # Shape: ()
        # debug.print("mean_counts: {mean_counts}",mean_counts=mean_counts)
        # Area within boundary (total "negative area") for each time step
        area_within_boundary = mean_counts / negative_counts.shape[0]**2  # Shape: ()
        # debug.print("area: {area_within_boundary}",area_within_boundary=area_within_boundary)
        # debug.print("neg_counts shape: {shape}",shape=negative_counts.shape[0])
        return area_within_boundary
    
    def vel_energy_net(self, params, t, x, y):
        """Velocity energy loss over batch points (avoids full field OOM)."""
        # Just use the raw (non-vmap) nets over the batch
        u = vmap(self.u_net, in_axes=(None, 0, 0, 0))(params, t, x, y)
        v = vmap(self.v_net, in_axes=(None, 0, 0, 0))(params, t, x, y)
        phi = vmap(self.phi_net, in_axes=(None, 0, 0, 0))(params, t, x, y)

        vel_mag2 = u**2 + v**2
        H_phi = self.smoothed_heaviside(phi, self.config.hs_epsilon)
        # eps = 1e-6

        # Log penalty
        # uv_loss = jnp.mean(H_phi * jnp.log(1.0 / (vel_mag2 + eps)))

        # Inverse penalty
        # uv_loss = jnp.mean(H_phi / (vel_mag2 + eps))

        # Treshold relu penalty
        v_target = 1.0  # Expected/desired minimum speed
        uv_loss = jnp.mean(H_phi * jnp.maximum(v_target - jnp.sqrt(vel_mag2), 0.0))

        return uv_loss
    
    @partial(jit, static_argnums=(0,))
    def res_and_w(self, params, batch, step):
        # Sort temporal coordinates
        t_sorted = batch[:, 0].sort()
        r_phi_pred, r_mom_x_pred, r_mom_y_pred, r_cont_pred = self.r_pred_fn(params, t_sorted, batch[:, 1], batch[:, 2], step)

        r_phi_pred =    r_phi_pred.reshape(self.num_chunks, -1)
        r_mom_x_pred =  r_mom_x_pred.reshape(self.num_chunks, -1)
        r_mom_y_pred =  r_mom_y_pred.reshape(self.num_chunks, -1)
        r_cont_pred =   r_cont_pred.reshape(self.num_chunks, -1)

        r_phi_l =       jnp.mean(r_phi_pred**2, axis=1)
        r_mom_x_l =     jnp.mean(r_mom_x_pred**2, axis=1)
        r_mom_y_l =     jnp.mean(r_mom_y_pred**2, axis=1)
        r_cont_l =      jnp.mean(r_cont_pred**2, axis=1)

        r_phi_gamma =       lax.stop_gradient(jnp.exp(-self.tol * (self.M @ r_phi_l)))
        r_mom_x_gamma =     lax.stop_gradient(jnp.exp(-self.tol * (self.M @ r_mom_x_l)))
        r_mom_y_gamma =     lax.stop_gradient(jnp.exp(-self.tol * (self.M @ r_mom_y_l)))
        r_cont_gamma =      lax.stop_gradient(jnp.exp(-self.tol * (self.M @ r_cont_l)))

        # Take minimum of the causal weights
        gamma = jnp.vstack([r_phi_gamma, r_mom_x_gamma, r_mom_y_gamma, r_cont_gamma])
        gamma = gamma.min(0)

        return r_phi_l, r_mom_x_l, r_mom_y_l, r_cont_l, gamma

    @partial(jit, static_argnums=(0,))
    def losses(self, params, batch, step):
        # Initial conditions
        phi0_pred = self.phi0_pred_fn(params, 0.0, self.x_star, self.y_star)
        p0_pred = self.p0_pred_fn(params, 0.0, self.x_star, self.y_star)
        u0_pred = self.u0_pred_fn(params, 0.0, self.x_star, self.y_star)
        v0_pred = self.v0_pred_fn(params, 0.0, self.x_star, self.y_star)
        
        ic_phi_loss = jnp.mean((phi0_pred - self.phi0)**2)
        ic_p_loss = jnp.mean((p0_pred - self.p0)**2)
        ic_u_loss = jnp.mean((u0_pred - self.u0)**2)
        ic_v_loss = jnp.mean((v0_pred - self.v0)**2)

        # BC loss
        # p_bc_pred = self.p_bc_pred_fn(params, self.bc_coords[:, 0], self.bc_coords[:, 1], self.bc_coords[:, 2])
        # p_bc_loss = jnp.mean((p_bc_pred - self.bc_values[:, 0]) ** 2)

        # Boundary conditions (no-slip walls)
        bc_u_pred = self.u_bc_pred_fn(
            params, 
            self.bc_coords["u=0"][:, 0],
            self.bc_coords["u=0"][:, 1],
            self.bc_coords["u=0"][:, 2],
        )
        bc_v_pred = self.v_bc_pred_fn(
            params, 
            self.bc_coords["v=0"][:, 0],
            self.bc_coords["v=0"][:, 1],
            self.bc_coords["v=0"][:, 2],
        )
        bc_p_pred = self.p_bc_pred_fn(
            params, 
            self.bc_coords["p=0"][:, 0],
            self.bc_coords["p=0"][:, 1],
            self.bc_coords["p=0"][:, 2],
        )
        bc_loss = jnp.mean(bc_u_pred**2 + bc_v_pred**2) + jnp.mean(bc_p_pred**2)

        # Eikonal loss
        if 'eik_p' in self.config.weighting.init_weights:
            eik_pred = self.eik_pred_fn(params, batch[:, 0], batch[:, 1], batch[:, 2])
            eik_loss = jnp.mean((eik_pred) ** 2)

        # Mass Loss (difference between computed area and exact mass)
        if 'mass_p' in self.config.weighting.init_weights:
            batch_ratio = int(self.config.training.batch_size_per_device / 16)
            mass_pred = self.mass_net(params, batch[::batch_ratio, 0], batch[::batch_ratio, 1], batch[::batch_ratio, 2])
            mass_loss = jnp.mean((mass_pred - self.exact_mass) ** 2)
            # debug.print("t: {t}",t=batch[::20, 0])

        # Velocity Energy Loss
        if 'uv_energy' in self.config.weighting.init_weights:
            decay_start = self.config.weighting.uv_start
            decay_duration = self.config.weighting.uv_duration
            decay_end = decay_start + decay_duration
            decay_rate = self.config.weighting.uv_decay_rate

            step_f = step.astype(jnp.float32)
            in_window = (step_f >= decay_start) & (step_f <= decay_end)

            # Define the decayed weight only when in the active window
            decay_weight = decay_rate * jnp.exp(-(step_f - decay_start) / decay_duration)
            energy_weight = jnp.where(in_window, decay_weight, 0.0)

            def compute_energy(_):
                return self.vel_energy_net(params, batch[:, 0], batch[:, 1], batch[:, 2])

            # Only compute vel_energy_net if in decay window
            raw_energy_loss = lax.cond(
                in_window,
                compute_energy,
                lambda _: 0.0,
                operand=None,
            )

            uv_energy_loss = energy_weight * raw_energy_loss

            # debug.print("step: {step}, decay_weight: {decay_weight}, uv_energy_loss: {uv_energy_loss}",
            #             step=step, decay_weight=decay_weight, uv_energy_loss=uv_energy_loss)

        # Residual loss
        if self.config.weighting.use_causal == True:
            # batch (64,3) uniform (spatial batch, t/x/y)
            res_batch = jnp.array([batch[:, 0].sort(), batch[:, 1], batch[:, 2]]).T   # (64,3)

            r_phi_l, r_mom_x_l, r_mom_y_l, r_cont_l, gamma = self.res_and_w(params, res_batch, step)
            r_phi_loss = jnp.mean(r_phi_l * gamma)
            r_mom_x_loss = jnp.mean(r_mom_x_l * gamma)
            r_mom_y_loss = jnp.mean(r_mom_y_l * gamma)
            r_cont_loss = jnp.mean(r_cont_l * gamma)

        else:
            r_phi_pred, r_mom_x_pred, r_mom_y_pred, r_cont_pred = self.r_pred_fn(
                params, batch[:, 0], batch[:, 1], batch[:, 2], step
            )
            # Compute loss
            r_phi_loss = jnp.mean(r_phi_pred**2)
            r_mom_x_loss = jnp.mean(r_mom_x_pred**2)
            r_mom_y_loss = jnp.mean(r_mom_y_pred**2)
            r_cont_loss = jnp.mean(r_cont_pred**2)

        loss_dict = {
            "ic_phi": ic_phi_loss,
            "ic_p": ic_p_loss,
            "ic_u": ic_u_loss,
            "ic_v": ic_v_loss,
            "bc": bc_loss,
            "r_phi": r_phi_loss,
            "r_mom_x": r_mom_x_loss,
            "r_mom_y": r_mom_y_loss,
            "r_cont": r_cont_loss,
           }
        if 'eik_p' in self.config.weighting.init_weights:
            loss_dict['eik_p'] = eik_loss
        if 'mass_p' in self.config.weighting.init_weights:
            loss_dict['mass_p'] = mass_loss
        if 'uv_energy' in self.config.weighting.init_weights:
            loss_dict['uv_energy'] = uv_energy_loss
        return loss_dict

    @partial(jit, static_argnums=(0,))
    def compute_l2_error(self, params, t, x, y, phi_ref, p_ref, u_ref, v_ref):
        phi_pred = self.phi_pred_fn(params, t, x, y)
        p_pred = self.p_pred_fn(params, t, x, y)
        u_pred = self.u_pred_fn(params, t, x, y)
        v_pred = self.v_pred_fn(params, t, x, y)

        phi_error = jnp.linalg.norm(phi_pred - phi_ref) / jnp.linalg.norm(phi_ref)
        p_error = jnp.linalg.norm(p_pred - p_ref) / jnp.linalg.norm(p_ref)
        u_error = jnp.linalg.norm(u_pred - u_ref) / jnp.linalg.norm(u_ref)
        v_error = jnp.linalg.norm(v_pred - v_ref) / jnp.linalg.norm(v_ref)

        return phi_error, p_error, u_error, v_error


class LevelSetEvaluator(BaseEvaluator):
    def __init__(self, config, model):
        super().__init__(config, model)

    def log_errors(self, params, phi_ref, p_ref, u_ref, v_ref):
        # Compute the L2 errors for phi
        phi_error, p_error, u_error, v_error = self.model.compute_l2_error(params, self.model.t_star, self.model.x_star, self.model.y_star, phi_ref, p_ref, u_ref, v_ref)
        self.log_dict["l2_phi_error"] = phi_error
        self.log_dict["l2_p_error"] = p_error
        self.log_dict["l2_u_error"] = u_error
        self.log_dict["l2_v_error"] = v_error

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

    def log_phi_rho(self, params):
        # Compute the value of phi in water and air
        phi_1 = self.model.phi_pred_fn(params, self.model.t_star, self.model.x_star, self.model.y_star) # Water
        phi_2 = self.model.phi_pred_fn(params, self.model.t_star, self.model.x_star, self.model.y_star) # Air
        phi_1 = phi_1[0,5,5]
        phi_2 = phi_2[0,-5,-5]
        self.log_dict["phi_1"] = phi_1
        self.log_dict["phi_2"] = phi_2

        # Compute the value of rho in water and air
        eps = self.model.config.hs_epsilon
        H_1 = 0.5 * (1 + jnp.tanh(phi_1 / eps))
        H_2 = 0.5 * (1 + jnp.tanh(phi_2 / eps))
        rho_1 = (1 - H_1) + self.model.rho_ratio * H_1
        rho_2 = (1 - H_2) + self.model.rho_ratio * H_2
        self.log_dict["rho_1"] = rho_1
        self.log_dict["rho_2"] = rho_2

    def __call__(self, state, batch, phi_ref, p_ref, u_ref, v_ref):
        # Initialize the log dictionary
        self.log_dict = super().__call__(state, batch)

        # Log causal weight if causal weighting is enabled
        if self.config.weighting.use_causal:
            _, _, _, _, causal_weight = self.model.res_and_w(state.params, batch, state.step)
            self.log_dict["cas_weight"] = causal_weight.min()

        # Log errors for p if enabled in the config
        if self.config.logging.log_errors:
            self.log_errors(state.params, phi_ref, p_ref, u_ref, v_ref)

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

        # Compute total loss
        total_loss = self.model.loss(state.params, state.weights, batch, state.step)
        self.log_dict["total_loss"] = total_loss

        # Log values of phi and rho in water and air
        if self.config.logging.log_phi_rho:
            self.log_phi_rho(state.params)

        return self.log_dict