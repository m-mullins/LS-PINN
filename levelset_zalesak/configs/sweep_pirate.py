import ml_collections

import jax.numpy as jnp


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    config.mode = "train_eval_s2s"   # "train" or "eval" or "train_eval" or "train_eval_s2s"
    config.dataset = "zalesak_nx101_nt41.npy"
    # config.dataset = "zalesak_nx21_nt41.npy"
    config.multi = False     # Add second level for refinement of eikonal or mass terms
    
    # Weights & Biases
    config.wandb = wandb = ml_collections.ConfigDict()
    wandb.project = "PINN-Level-set-Zalesak"
    wandb.name = "sweep_pirate"
    wandb.tag = ["zalesak"]
    wandb.notes = "sweep"

    # Nondimensionalization
    config.nondim = nondim = ml_collections.ConfigDict()
    nondim.nondimensionalize = False
    nondim.X_star = 1.0
    nondim.Y_star = 1.0
    nondim.P_star = 1.0
    nondim.V_star = 1.0
    nondim.T_star = 2.0

    # Circular mask to account for circular region of zalesak FEM reference solution instead of unit square
    config.mask = mask = ml_collections.ConfigDict()
    mask.center_x = 0.5
    mask.center_y = 0.5
    mask.radius = 0.45
    
    # Physics-informed initialization
    config.use_pi_init = True
    config.pi_init_type = "initial_condition"   # "linear_pde" or "initial_condition"

    # Arch
    config.arch = arch = ml_collections.ConfigDict()
    arch.arch_name = "PirateNet"
    arch.num_layers = 3
    arch.hidden_dim = 256
    arch.out_dim = 1
    arch.activation = "gelu" # "tanh"
    arch.periodicity = False
    arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": 1, "embed_dim": 256})
    arch.reparam = ml_collections.ConfigDict(
        {"type": "weight_fact", "mean": 0.5, "stddev": 0.1}     # RWF
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

    # Optim
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "Adam"
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
    training.max_steps = 80000 # 20000
    training.batch_size_per_device = 2048 # 2048
    training.s2s = True                # Sequence to sequence learning
    training.num_time_windows = 1    # For seq2seq

    # Weighting of loss terms
    config.weighting = weighting = ml_collections.ConfigDict()
    weighting.scheme = "grad_norm"  # "grad_norm" or "ntk" or None
    # weighting.init_weights = ml_collections.ConfigDict({"ic_p": 5.0, "p_bc": 0.5, "rp": 10.0, "eik_p": 0.0000001, "mass_p": 0.1}) # lambda
    weighting.init_weights = ml_collections.ConfigDict({"ic_p": 1.0, "rp": 1.0}) # lambda
    weighting.momentum = 0.9
    weighting.update_every_steps = 500

    weighting.use_causal = True # Respecting Temporal Causality algorithm
    weighting.causal_tol = 1.0  # epsilon
    weighting.num_chunks = 32   # number of subdivisions

    # Training second level
    config.training1 = training1 = ml_collections.ConfigDict()
    training1.max_steps = 1000 # 1000
    training1.batch_size_per_device = 512 # 1024
    training1.log_every_steps = 50 # 50

    # Weighting of loss terms for second level
    config.weighting1 = weighting1 = ml_collections.ConfigDict()
    weighting1.scheme = "grad_norm"  # "grad_norm" or "ntk" or None
    # weighting1.init_weights = ml_collections.ConfigDict({"eik_p": 1.0, "mass_p": 1.0}) # lambda
    weighting1.init_weights = ml_collections.ConfigDict({"mass_p": 1.0}) # lambda
    weighting1.momentum = 0.9
    weighting1.update_every_steps = 500

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

    # Saving
    config.saving = saving = ml_collections.ConfigDict()
    saving.save_every_steps = 20000 # 10000
    saving.num_keep_ckpts = 10

    # Input shape for initializing Flax models (amount of dim in domain)
    config.input_dim = 3

    # Integer for PRNG random seed.
    config.seed = 42

    return config
