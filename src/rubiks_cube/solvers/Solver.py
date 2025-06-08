from abc import ABC, abstractmethod
import numpy as np
import time
from pathlib import Path
import threading

class Solver(ABC):
    @abstractmethod
    def solve_cube(self, cube):
        """
        
        """