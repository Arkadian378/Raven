"""LabyrinthRL: a multi-agent POMDP environment for Ravensburger's Labyrinth."""
from .core import NUM_ACTIONS, GameState
from .env import LabyrinthEnv, RewardConfig
from .observation import ObservationConfig

__all__ = ["GameState", "LabyrinthEnv", "RewardConfig", "ObservationConfig", "NUM_ACTIONS"]
__version__ = "0.1.0"
