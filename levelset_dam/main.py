import os
import sys

# Deterministic
# os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # DETERMINISTIC
os.environ["WANDB__SERVICE_WAIT"] = "120"

from absl import app
from absl import flags
from absl import logging

from ml_collections import config_flags

import jax
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
# Add project root path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import train
import eval

FLAGS = flags.FLAGS

workdir = os.path.dirname(os.path.abspath(__file__))
flags.DEFINE_string("workdir", workdir, "Directory to store model data.")

config_flags.DEFINE_config_file(
    "config",
    "./configs/default.py",
    # "./configs/default_curri.py",
    # "./configs/pirate.py",
    # "./configs/curri.py",
    # "./configs/pirate_curri_2.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

def main(argv):
    if FLAGS.config.mode == "eval_s2s":       # Regular evalutation of results with sequence to sequence
        eval.evaluate_s2s(FLAGS.config, FLAGS.workdir)
        eval.plot_mass_loss(FLAGS.config, FLAGS.workdir)
        eval.plot_sliced_results(FLAGS.config, FLAGS.workdir)
        eval.plot_heaviside_and_density(FLAGS.config, FLAGS.workdir)
    
    elif FLAGS.config.mode == "train_eval_s2s": # Train and then evaluate the results with sequence to sequence
        train.train_and_evaluate_s2s(FLAGS.config, FLAGS.workdir)
        eval.evaluate_s2s(FLAGS.config, FLAGS.workdir)
        eval.plot_mass_loss(FLAGS.config, FLAGS.workdir)
        eval.plot_sliced_results(FLAGS.config, FLAGS.workdir)
        eval.plot_heaviside_and_density(FLAGS.config, FLAGS.workdir)
        eval.evaluate_s2s_reinit(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "curri":      # Implement curriculum learning scheme from Krishnapriyan et. al. (2021)
        datasets = FLAGS.config.transfer.datasets
        iterations = FLAGS.config.transfer.iterations
        rho1s = FLAGS.config.transfer.rho1s
        rho2s = FLAGS.config.transfer.rho2s
        mu1s = FLAGS.config.transfer.mu1s
        mu2s = FLAGS.config.transfer.mu2s

        FLAGS.config.logging.global_step = 0
        for dataset, iteration, rho1, rho2, mu1, mu2 in zip(datasets, iterations, rho1s, rho2s, mu1s, mu2s):
            # Update the config to use the current dataset and amount of iterations
            FLAGS.config.dataset = dataset
            FLAGS.config.training.max_steps = iteration

            # Calculate properties
            FLAGS.config.rho1 = rho1
            FLAGS.config.rho2 = rho2
            FLAGS.config.mu1 = mu1
            FLAGS.config.mu2 = mu2
            FLAGS.config.nondim.P_star = rho1 * np.abs(FLAGS.config.g) * FLAGS.config.nondim.X_star
            FLAGS.config.nondim.Re = rho1 * ((np.abs(FLAGS.config.g) * FLAGS.config.nondim.X_star) ** 0.5) * FLAGS.config.nondim.X_star / mu1
            FLAGS.config.nondim.rho_ratio = rho2 / rho1
            FLAGS.config.nondim.mu_ratio = mu2 / mu1

            logging.info(f"Training and evaluating with dataset: {dataset} for {iteration} iterations")
            logging.info(f"Fluid properties: rho1={rho1} | rho2={rho2} | mu1={mu1} | mu2={mu2}")
            logging.info(f"Fluid properties: rho_ratio={FLAGS.config.nondim.rho_ratio} | mu_ratio={FLAGS.config.nondim.mu_ratio}")
            logging.info(f"Fluid properties: P_star={FLAGS.config.nondim.P_star} | Re={FLAGS.config.nondim.Re}")

            train.train_and_evaluate_s2s(FLAGS.config, FLAGS.workdir)
            eval.evaluate_s2s(FLAGS.config, FLAGS.workdir)
            eval.plot_mass_loss(FLAGS.config, FLAGS.workdir)
            eval.plot_sliced_results(FLAGS.config, FLAGS.workdir)
            eval.plot_heaviside_and_density(FLAGS.config, FLAGS.workdir)
            
            FLAGS.config.logging.global_step = FLAGS.config.logging.global_step + FLAGS.config.training.max_steps

            FLAGS.config.use_pi_init = False                # Remove pi init for following datasets
            FLAGS.config.transfer.s2s_pi_init = False       # Remove pi init for following datasets
            FLAGS.config.transfer.curri_step = iteration    # Update iteration from which the next dataset will init it's params and state
    
    elif FLAGS.config.mode == "eval_sliced": # Plot the static results for n time steps for paper
        eval.plot_sliced_results(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_mass": # Plot the static results for n time steps for paper
        eval.plot_mass_loss(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_reinit": # Plot the static results for n time steps for paper
        eval.evaluate_s2s_reinit(FLAGS.config, FLAGS.workdir)
    
if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
