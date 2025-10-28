import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train_eval_s2s"   # "train" or "eval" or "train_eval" or "train_eval_s2s"
    config.dataset = "dam_nx41_nt50_t0.25.npy"
    # config.dataset = "dam_nx41_nt100.npy"
    config.rho1 = 1000          # Density (kg/m3) | Water
    config.rho2 = 1             # Density (kg/m3) | Air
    config.mu1 = 0.1          # Dynamic viscosity (Pa.s) | Water
    config.mu2 = 0.001825     # Dynamic viscosity (Pa.s) | Air
    config.g = -9.81            # Gravity (m/s2)
    config.hs_epsilon = 0.01    # Smoothed heavisde epsilon (higher = less steep)
    
    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Level-set-dam"
    wandb.name = "pirate_curri"
    wandb.tag = []
    wandb.notes = "increase visc*100, re_schedule step=10, four=2, 10k, s2s=25, bs=2048"

    # Nondimensionalization
    config.nondim = nondim = ml_collections.ConfigDict()
    nondim.nondimensionalize = True
    nondim.X_star = 0.584                   # L
    nondim.Y_star = 0.584                   # L
    nondim.T_star = (0.584 / 9.81) ** 0.5   # sqrt(L/g)
    nondim.PHI_star = 1.0
    nondim.P_star = 1000 * 9.81 * 0.584     # rho_ref * g * L
    nondim.U_star = (9.81 * 0.584) ** 0.5   # sqrt(g*L)
    nondim.V_star = (9.81 * 0.584) ** 0.5   # sqrt(g*L)
    nondim.Re = 1000 * ((9.81 * 0.584) ** 0.5) * 0.584 / 0.1 # Re = rho_ref * sqrt(gL) * L / mu_ref
    nondim.rho_ratio = 1 / 1000             # rho_air/rho_water
    nondim.mu_ratio = 0.001825 / 0.1    # mu_air/mu_water

    # Physics-informed initialization
    config.use_pi_init = True
    config.pi_init_type = "initial_condition"   # "initial_condition"

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 3
    arch.hidden_dim = 256
    arch.out_dim = 4
    arch.activation = "tanh" # "tanh"
    arch.periodicity = False
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 2, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 1.0, "stddev": 0.1}     # RWF
    )
    arch.nonlinearity = 0.0 # alpha
    arch.pi_init = None # Leave as none, is updated with weights in train script

    # Transfer learning
    config.transfer = transfer = ml_collections.ConfigDict()
    transfer.curriculum = False # Curriculum learning scheme
    transfer.datasets = None    # List of dataset filenames for curriculum training
    transfer.iterations = None  # List of training iterations for each dataset
    transfer.curri_step = None  # Leave as none. Iteration from which init state will be passed for curriculum learning
    transfer.s2s_transfer = True # Use transfer learning to initiate params of subsequent time windows in seq-2-seq learning
    transfer.s2s_pi_init = True # Leave as True if s2s_transfer is also True. Will change to false after first window.
    transfer.rho1s = [1000., 998.]  # List of densities for each dataset
    transfer.rho2s = [500., 1.2]    # List of densities for each dataset
    transfer.mu1s = [0.03, 0.04]   # List of viscosities for each dataset
    transfer.mu2s = [0.01, 0.02]   # List of viscosities for each dataset
    
    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Soap"    # "Adam" or "Soap"
    optim.beta1 = 0.9
    optim.beta2 = 0.999
    optim.eps = 1e-8
    optim.learning_rate = 1e-3
    optim.decay_rate = 0.9
    optim.decay_steps = 2000
    optim.staircase = False
    optim.warmup_steps = 2000
    optim.grad_accum_steps = 0

    # Training
    config.training = training = ml_collections.ConfigDict()
    training.max_steps = 10000 # 50000
    training.batch_size_per_device = 2048 # 2048
    training.s2s = True                # Sequence to sequence learning
    training.num_time_windows = 25    # For seq2seq
    training.grad_clip = 2.0   # Non dimensional gradient clipping min/max
    training.re_schedule = "step"   # Increase re during training according to a schedule "step", "sigmoid" or None
    training.re_schedule_n = 10      # Number of steps in the step re schedule
    training.re_schedule_k = 10     # Steepness of the sigmoid curve for re schedule
    training.re_min = 100           # Min value of re for the schedule
    training.reinit = True  # Reinitialize the level set at each time window in seq2seq learning
    training.reinit_refine_scale = 4  # Refine level set better reinit
    
    # Weighting of loss terms
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"  # "grad_norm" or "ntk" or None
    weighting.init_weights = ml_collections.ConfigDict({
        "ic_phi": 1.0, 
        "ic_p": 1.0, 
        "ic_u": 1.0, 
        "ic_v": 1.0, 
        "bc": 1.0, 
        "r_phi": 1.0, 
        "r_mom_x": 1.0, 
        "r_mom_y": 1.0, 
        "r_cont": 1.0,
        # "eik_p": 0.0001,
    }) # lambda
    weighting.momentum = 0.9
    weighting.update_every_steps = 500

    weighting.use_causal = True # Respecting Temporal Causality algorithm
    weighting.causal_tol = 1.0  # epsilon
    weighting.num_chunks = 32   # number of subdivisions
    weighting.uv_start = 5000   # Iteration start for uv_energy in loss
    weighting.uv_duration = 10000   # Amount of iterations to use uv_energy in loss
    weighting.uv_decay_rate = 0.01   # uv energy decay rate
    
    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.log_every_steps = 500   # 500
    logging.global_step = None      # Leave as none, updated automatically in train with curriculum
    logging.log_errors = True
    logging.log_losses = True
    logging.log_weights = True
    logging.log_nonlinearities = True
    logging.log_preds = False
    logging.log_grads = False
    logging.log_ntk = False
    logging.log_phi_rho = True

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 20000 # 20000
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models (amount of dim in domain)
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
