from omegaconf import DictConfig
import torch.optim as optim
from models.sceneRepresentation import Scene


class optimizersScene():
    def __init__(self, model: Scene, cfg_optimizers: DictConfig, only_bg=False):
        params = model.get_params()

        self.optimizers = {}

        # Optimizer for background representation
        self.optimizers['bg_repr'] = optim.Adam(
            params["parameter_background"]["params_repr"],
            lr=cfg_optimizers.background.lr_repr
        )

        if not only_bg:
            # Optimizer for local representation
            self.optimizers['obj_repr'] = optim.Adam(
                params["parameter_obj"]["params_repr"],
                lr=cfg_optimizers.object.lr_repr
            )

            # Optimizer for physical parameters
            self.optimizers['obj_physics'] = optim.Adam(
                params["parameter_obj"]["params_physics"],
                lr=cfg_optimizers.physics.lr_physics
            )

            self.optimizers['homography'] = optim.Adam(
                [model.homography_matrix],
                lr=cfg_optimizers.physics.lr_physics
            )

        # Learning Rate Functions
        def lr_lambda_bg_repr(epoch):
            return cfg_optimizers.background.lr_scheduler_gamma_repr ** (
                epoch / cfg_optimizers.background.lr_scheduler_step_size_repr
            )

        if not only_bg:
            def lr_lambda_object_repr(epoch):
                return cfg_optimizers.object.lr_scheduler_gamma_repr ** (
                    epoch / cfg_optimizers.object.lr_scheduler_step_size_repr
                )

            def lr_lambda_object_physics(epoch):
                return cfg_optimizers.physics.lr_scheduler_gamma_physics ** (
                    epoch / cfg_optimizers.physics.lr_scheduler_step_size_physics
                )

        # Learning rate scheduler
        self.lr_scheduler = {}
        self.lr_scheduler['bg_repr'] = optim.lr_scheduler.LambdaLR(
            self.optimizers['bg_repr'], lr_lambda_bg_repr, last_epoch=-1, verbose=False
        )

        if not only_bg:
            self.lr_scheduler['obj_repr'] = optim.lr_scheduler.LambdaLR(
                self.optimizers['obj_repr'], lr_lambda_object_repr, last_epoch=-1, verbose=False
            )
            self.lr_scheduler['obj_physics'] = optim.lr_scheduler.LambdaLR(
                self.optimizers['obj_physics'], lr_lambda_object_physics, last_epoch=-1, verbose=False
            )

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad()

    def optimizer_step(self):
        for optimizer in self.optimizers.values():
            optimizer.step()

    def lr_scheduler_step(self):
        for scheduler in self.lr_scheduler.values():
            scheduler.step()
