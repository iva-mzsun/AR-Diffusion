from accelerate.tracking import GeneralTracker, on_main_process
from typing import Optional

import wandb


class WandBTracker_wImage(GeneralTracker):
    name = "wandb"
    requires_logging_directory = False

    @on_main_process
    def __init__(self, run_name: str):
        self.run_name = run_name
        run = wandb.init(self.run_name, dir='./experiments/wandb')

    @property
    def tracker(self):
        return self.run.run

    @on_main_process
    def store_init_configuration(self, values: dict):
        wandb.config(values)

    @on_main_process
    def log(self, values: dict, step: Optional[int] = None):
        wandb.log(values, step=step)
