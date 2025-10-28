# LS-PINN

LS-PINN is a PINN solver for solving level set-based problems (LS) developed at École de technologie supérieure within the group [GRANIT](https://www.etsmtl.ca/recherche/laboratoires-et-chaires-ets/granit) by Mathieu Mullins.

The code was built with [JAX](https://docs.jax.dev/) on top of the existing [JAX-PI](https://github.com/PredictiveIntelligenceLab/jaxpi/) code.

If using the code, please reference the article:

Mullins, M., Kamil, H., Fahsi, A., & Soulaïmani, A. (2025). Physics-informed neural networks for solving moving interface flow problems using the level set approach. Physics of Fluids, 37(10), 107124. https://doi.org/10.1063/5.0289386


## Abstract

This paper advances the use of physics-informed neural networks (PINNs) architectures to address moving interface problems via the level set method. Originally developed for other partial differential equations-based problems, we particularly leverage physics-informed deep learning with residual adaptive networks' (PirateNet) features—including causal training, sequence-to-sequence learning, random weight factorization, and Fourier feature embeddings—and tailor them to handle problems with complex interface dynamics. Numerical experiments validate this framework on benchmark problems such as Zalesak's disk rotation and time-reversed vortex flow. We demonstrate that PINNs can efficiently solve level set problems exhibiting significant interface deformation without the need for upwind numerical stabilization, as generally required by classic discretization methods, or additional mass conservation schemes. However, incorporating an Eikonal regularization term in the loss function with an appropriate weight can further enhance results in specific scenarios. Our results indicate that PINNs with the PirateNet architecture surpass conventional PINNs in accuracy, achieving state-of-the-art error rates of $L^2=0.14\%$ for Zalesak's disk and $L^2=0.85\%$ for the time-reversed vortex flow problem, as compared to reference solutions. Additionally, for a complex two-phase flow dam break problem coupling the level set with the Navier–Stokes equations, we propose a geometric reinitialization method embedded within the sequence-to-sequence training scheme to ensure long-term stability and accurate inference of the level set field. The proposed framework has the potential to be broadly applicable to industrial problems that involve moving interfaces, such as free-surface flows in hydraulics and maritime engineering.

## Quickstart

This code was built for GPU usage on HPC. Our code was tested on the Digital Research Alliance of Canada's [Beluga](https://docs.alliancecan.ca/wiki/B%C3%A9luga/) and [Cedar](https://docs.alliancecan.ca/wiki/Cedar) clusters with 1 GPU.
It was tested with the following versions:

- Python 3.11.5
- JAX 0.4.30
- CUDA 12.2

The code uses [Weights & Biases](https://wandb.ai/site) to log and monitor training metrics. Before starting, make sure you have a working WandB account with the API key.

Here are the steps to follow before starting a training run:

Once logged into the cluster, move to the desired folder and load the proper modules:
```
cd ~/projects/group/account/
module load StdEnv/2023 python/3.11.5 cuda/12.2
```

Clone the repo, preferably using SSH:
```
git clone git@github.com:m-mullins/LS-PINN.git
```

Create and activate a virtual environment:
```
virtualenv env
source env/bin/activate
pip install--no-index--upgrade pip
```

Install requirements:
```
cd LS-PINN
pip install -r requirements.txt
```

Enter WandB API Key:
```
WANDB__SERVICE_WAIT=300 wandb login API-KEY
```

## Starting a training run

Here are the steps to follow to start a training run. We will use the time-reversed vortex flow as an example (``levelset_cst``).

Modify the main.py to choose the configuration you want to train:
```
nano levelset_cst/main.py
```

If you want to run ``plain.py``, modify the config_flags to this:
```
config_flags.DEFINE_config_file(
    "config",
    "./configs/plain.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)
```

To start training:
```
sbatch scripts/train_ls_vortex.sh
```

Since most of DRAC's clusters are offline, all training runs are offline runs for WandB tracking. You can monitor when the job ends with `sq`. Once training is over, the outfile will contain the line needed to sync the run to your WandB account.
```
nano outfiles/ls.out
```
The line to copy should be at the very end, it begins with `wandb sync`. Paste the line in the terminal:
```
wandb sync /project/1234567/account/LS-PINN/levelset_cst/wandb/offline-run-YYYYMMDD_123456789
```
After a few moments, the run will be available on your WandB account.

## Results

### Time-reversed vortex flow
![levelset_cst](examples/time_rev_vortex_pirate_s2s.gif)

### Dam break with reinit
![levelset_dam](examples/level_set_dam_break_reinit.gif)