# Propulate_Demo
Short demo for Hyperparameter search with Propulate.

## Setup on LUMI
- [ ] `module use /appl/local/csc/modulefiles/`
- [ ] `module load pytorch/2.5`
- [ ] `python -m venv pvenv --system-site-packages`
- [ ] `source pvenv/bin/activate`
- [ ] `git clone https://github.com/Helmholtz-AI-Energy/propulate.git`
- [ ] `cd propulate`
- [ ] `pip install -e .`
- [ ] `cd ../Propulate_Tutorial/`

## Toy Example
See `toy.py` and `run_toy.sh`.

## Neural Network Example
See `nas.py` and `run_nas.sh`.

## Multi-Rank Worker Example
See `ddp.py` and `run_ddp.sh`
