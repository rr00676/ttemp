from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

@dataclass
class ExperimentConfig:
    """Configuration for simulation experiments."""
    time: float  # Total simulation time
    hz: float  # Sampling frequency
    trials: int  # Number of simulation trials
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        self.dt = 1.0 / self.hz  # Time step
        self.steps = int(self.time * self.hz)  # Total number of steps
        if self.metadata is None:
            self.metadata = {}
        
    @property
    def time_points(self) -> np.ndarray:
        """Return array of time points for the simulation."""
        return np.linspace(0, self.time, self.steps + 1)