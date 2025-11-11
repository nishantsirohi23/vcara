"""V-CARA: uncertainty-gated counterfactual RL for visual control."""
from .core import (Cfg, VARIANTS, make_cfg, VCARAAgent, CarRacingPixels,
                   EpisodeReplay, evaluate, train)

__version__ = "0.3.1"
__all__ = ["Cfg", "VARIANTS", "make_cfg", "VCARAAgent", "CarRacingPixels",
           "EpisodeReplay", "evaluate", "train", "__version__"]
