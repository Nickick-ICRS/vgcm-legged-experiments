from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import os

from legged_gym import LEGGED_GYM_ROOT_DIR

class ExperimentBase(ABC):
    def __init__(self, sim, duration: float):
        """
        Abstract base class for running experiments in a MuJoCo simulation.
        
        :param sim: The MuJoCo simulation instance
        :param experiment_name: Name of the experiment (used for saving data)
        :param duration: Experiment duration in simulation time
        """
        self.sim = sim
        self.duration = duration
        self.n_robots = sim.num_robots
        self.sim.set_test_duration(duration)
        self.sim.register_callback(self.update)
        self.prepare_sim(self.sim)
        self.state_histories = [
            pd.DataFrame(np.nan,
                         index=np.arange(sim.steps),
                         columns=sim.get_full_state_dict(i).keys())
            for i in range(self.n_robots)
        ]
        self.current_step = 0

    def update_history(self):
        """Add new data to the state history."""
        for i in range(self.n_robots):
            data = self.sim.get_full_state_dict(i)
            self.state_histories[i].loc[self.current_step] = data
    
    def is_done(self) -> bool:
        self.current_step += 1
        if self.current_step % 1000 == 0:
            print(f"Step {self.current_step} / {self.sim.steps}")
        """Check if the experiment duration has elapsed and signal termination."""
        if self.sim.steps > 0 and self.current_step > self.sim.steps:
            self.sim.signal_shutdown()
            self.finish()
            return True
        return False
    
    def save_results(self):
        """Save state history to a JSON file."""
        filepath = os.path.join(LEGGED_GYM_ROOT_DIR, "vgcm/experiment_results")
        for i, df in enumerate(self.state_histories):
            filename = self.get_filename(i)
            path = os.path.join(filepath, filename)
            df.dropna()
            df.to_csv(path, index=False)
    
    @abstractmethod
    def update(self) -> None:
        """Callback function called every simulation update."""
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Function called when the experiment finishes."""
        pass

    @abstractmethod
    def prepare_sim(self, sim) -> None:
        """
        Function called when initialised. Insert any custom simulation setup here. 
        """
        pass

    @abstractmethod
    def experiment_name(self) -> str:
        """
        Get the name of this experiment
        """
        pass

    def get_filename(self, idx) -> str:
        return f"{self.experiment_name()}_results_{idx}.json"
