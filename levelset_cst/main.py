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
    "./configs/default_s2s_vortex.py",
    # "./configs/plain_s2s_vortex.py",
    # "./configs/sota_vortex.py",
    # "./configs/pirate_s2s.py",
    # "./configs/pirate_s2s_vortex.py",
    # "./configs/sota_multi_vortex.py",
    # "./configs/pirate_s2s_linear.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

def main(argv):
    if FLAGS.config.mode == "eval_s2s":       # Regular evalutation of results with sequence to sequence
        eval.evaluate_s2s(FLAGS.config, FLAGS.workdir)
        eval.plot_mass_loss(FLAGS.config, FLAGS.workdir)
        eval.plot_sliced_results(FLAGS.config, FLAGS.workdir)
    
    elif FLAGS.config.mode == "train_eval_s2s": # Train and then evaluate the results with sequence to sequence
        train.train_and_evaluate_s2s(FLAGS.config, FLAGS.workdir)
        eval.evaluate_s2s(FLAGS.config, FLAGS.workdir)
        eval.plot_mass_loss(FLAGS.config, FLAGS.workdir)
        eval.plot_sliced_results(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "curri":      # Implement curriculum learning scheme from Krishnapriyan et. al. (2021)
        datasets = FLAGS.config.transfer.datasets
        iterations = FLAGS.config.transfer.iterations
        FLAGS.config.logging.global_step = 0
        for dataset, iteration in zip(datasets, iterations):
            FLAGS.config.dataset = dataset  # Update the config to use the current dataset
            FLAGS.config.training.max_steps = iteration  # Update the config to use the current dataset
            logging.info(f"Training and evaluating with dataset: {dataset} for {iteration} iterations")
            train.train_and_evaluate_s2s(FLAGS.config, FLAGS.workdir)
            eval.evaluate_s2s(FLAGS.config, FLAGS.workdir)
            FLAGS.config.logging.global_step = FLAGS.config.logging.global_step + FLAGS.config.training.max_steps

            FLAGS.config.use_pi_init = False # Remove pi init for following datasets
            FLAGS.config.transfer.s2s_pi_init = False   # Remove pi init for following datasets
            FLAGS.config.transfer.curri_step = iteration # Update iteration from which the next dataset will init it's params and state
    
    elif FLAGS.config.mode == "eval_sliced": # Plot the static results for n time steps for paper
        eval.plot_sliced_results(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "eval_mass": # Plot the static results for n time steps for paper
        eval.plot_mass_loss(FLAGS.config, FLAGS.workdir)

    elif FLAGS.config.mode == "train_eval_multi": # Train and then evaluate the results with sequence to sequence
        train.train_two_level(FLAGS.config, FLAGS.workdir)
        eval.evaluate_s2s(FLAGS.config, FLAGS.workdir)
        eval.plot_sliced_results(FLAGS.config, FLAGS.workdir)
        eval.plot_mass_loss(FLAGS.config, FLAGS.workdir)
    
if __name__ == "__main__":
    flags.mark_flags_as_required(["config", "workdir"])
    app.run(main)
