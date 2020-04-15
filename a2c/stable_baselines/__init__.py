# Load mpi4py-dependent algorithms only if mpi is installed.
try:
    import mpi4py
except ImportError:
    mpi4py = None

if mpi4py is not None:
    from stable_baselines.ddpg import DDPG
    from stable_baselines.gail import GAIL
    from stable_baselines.ppo1 import PPO1
    from stable_baselines.trpo_mpi import TRPO
del mpi4py

__version__ = "2.10.1a0"
