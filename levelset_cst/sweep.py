import os
import sys
import platform

# Deterministic
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC
os.environ["WANDB__SERVICE_WAIT"] = "120"

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags, config_dict
import ml_collections

import jax, wandb

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# Add project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from jaxpi.logging import Logger

import train
import eval

FLAGS = flags.FLAGS

workdir = os.path.dirname(os.path.abspath(__file__))
flags.DEFINE_string("workdir", workdir, "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    # "./configs/sweep_pirate.py",
    # "./configs/sweep_default.py",
    "./configs/sweep_plain.py",
    "File path to the training hyperparameter configuration.",
    # lock_config=True,
)

def main(argv):
    config = FLAGS.config
    workdir = FLAGS.workdir

    logger = Logger()
    logger.info(f"Sweep main script start")

    sweep_config = {
        "method": "bayes",
        "name": "sweep_pirate_ls",
        "metric": {"goal": "minimize", "name": "l2_p_error"},
        "early_terminate": {
            "type": "hyperband",  # Optional: early stopping with HyperBand strategy
            "min_iter": 5000,    # Min epochs for each trial before pruning
        },
    }

    parameters_dict = {
        "num_layers": {"values": [3, 4, 6, 8]},
        "activation": {"values": ["relu", "tanh", "gelu", "swish"]},
        # "fourier_scale": {"values": [1, 2]},
        # "rwf_mean": {"values": [0.5, 1.0]},
        # "causal_tol": {"values": [1.0, 1.5, 2.0]},
    }

    sweep_config["parameters"] = parameters_dict

    def train_sweep():
        # Initialize logger
        logger = Logger()

        config = FLAGS.config

        # Find out if running on pc for dubugging or on HPC without internet access
        if 'microsoft' in platform.uname().release.lower():
            mode = "online"
        else:
            mode = "offline"
        mode = "online" # On Cedar HPC with internet access

        logger.info(f"Initializing wandb sweep run")
        wandb.init(project=config.wandb.project, name=config.wandb.name, mode=mode)
        logger.info(f"wandb sweep run initialized {mode}")

        sweep_config = wandb.config

        # Update config with sweep parameters
        config.arch.num_layers = sweep_config.num_layers
        config.arch.activation = sweep_config.activation
        # config.arch.fourier_emb = ml_collections.ConfigDict({"embed_scale": sweep_config.fourier_scale, "embed_dim": 256})
        # config.arch.reparam = ml_collections.ConfigDict({"type": "weight_fact", "mean": sweep_config.rwf_mean, "stddev": 0.1})
        # config.weighting.causal_tol = sweep_config.causal_tol

        config.arch.pi_init = (
            None  # Reset pi_init every sweep otherwise it will be overwritten!!!
        )

        train.train_and_evaluate_s2s(config, workdir)
        eval.evaluate_s2s(config, workdir)

        config.transfer.s2s_pi_init = True

    logger.info(f"Initializing wandb sweep id")
    sweep_id = wandb.sweep(
        sweep_config,
        project=config.wandb.project,
    )

    logger.info(f"Initializing wandb sweep agent")
    wandb.agent(sweep_id, function=train_sweep)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)