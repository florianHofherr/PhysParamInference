from omegaconf import DictConfig
import torch


class Losses():
    def __init__(self, cfg_loss: DictConfig, **kwargs):
        """
        """
        self.criterion_color = torch.nn.MSELoss()

        self.losses_epoch = {
            'loss_full': 0,
            'loss_color': 0
        }

        if "weight_regularization_mask" in cfg_loss:
            self.regularize_mask = True
            self.weight_regularization_mask = cfg_loss.weight_regularization_mask

            def criterion_mask_regularization(mask):
                return torch.mean(mask * (1-mask))
            self.criterion_mask_regularization = criterion_mask_regularization

            self.losses_epoch['loss_mask_regularization'] = 0
        else:
            self.regularize_mask = False

        self.loss_for_backward = None

    def add_losses_spring(
        self,
        cfg_loss: DictConfig,
        traj_center1,
        traj_center2,
    ):
        """
        """
        self.weight_segmentation = cfg_loss.weight_segmentation
        self.criterion_segmentation = torch.nn.BCELoss()

        self.losses_epoch['loss_segmentation'] = 0

        self.weight_initial_pos = cfg_loss.weight_initial_pos
        self.criterion_initial_pos = torch.nn.MSELoss()

        self.traj_center1 = traj_center1
        self.traj_center2 = traj_center2

        self.losses_epoch['loss_initital_pos'] = 0

        self.weight_artefacts = cfg_loss.weight_artefacts
        self.artifact_loss_after = cfg_loss.activate_artifact_loss_after
        self.criterion_artefacts = torch.nn.MSELoss()
        self.losses_epoch['loss_artefacts'] = 0

    def compute_losses(
        self,
        output_model,
        true_colors
    ):
        rendered_colors = output_model['colors']
        loss_color = self.criterion_color(
            rendered_colors.reshape(-1, 3),
            true_colors.reshape(-1, 3)
        )

        loss_full = loss_color

        if self.regularize_mask:
            # Different regularization for spring example (two object masks) and the rest
            if not type(output_model['mask']) is dict:
                loss_mask_reg = self.weight_regularization_mask * self.criterion_mask_regularization(output_model['mask'])
            else:
                loss_mask_reg = self.weight_regularization_mask * self.criterion_mask_regularization(output_model['mask']['mask1']) +\
                    self.weight_regularization_mask * self.criterion_mask_regularization(output_model['mask']['mask2'])

            loss_full += loss_mask_reg
            self.losses_epoch['loss_mask_regularization'] += loss_mask_reg.detach().cpu()

        self.loss_for_backward = loss_full

        self.losses_epoch['loss_color'] += loss_color.detach().cpu()
        self.losses_epoch['loss_full'] += loss_full.detach().cpu()

    def compute_losses_spring(
        self,
        model,
        output_model,
        true_mask,
        coords,
        tspan,
        epoch,
    ):
        device = next(model.parameters()).device

        # Additional segmentation loss
        loss_segmentation = self.weight_segmentation * self.criterion_segmentation(
            output_model['mask']['mask1'][:, :1], true_mask['mask1'][:, :1].float()
        )
        loss_segmentation += self.weight_segmentation * self.criterion_segmentation(
            output_model['mask']['mask2'][:, :1], true_mask['mask2'][:, :1].float()
        )

        # Loss on the initial positions of the local representations
        loss_initital_pos = self.weight_initial_pos * self.criterion_initial_pos(
            model.local_representation.x0[:2], self.traj_center1[0]
        )
        loss_initital_pos += self.weight_initial_pos * self.criterion_initial_pos(
            model.local_representation.x0[2:4], self.traj_center2[0]
        )

        # Loss to reduce artifacts in the non-visible part during training, that might become visible
        # when extrapolating (evaluate mask at non-visible points and force to be zero)
        # Add bounding box to coordinates
        if epoch > self.artifact_loss_after:
            surrounding_points = torch.cat([
                    coords + torch.tensor([1.0, 0.0]).to(device),
                    coords + torch.tensor([1.0, 1.0]).to(device),
                    coords + torch.tensor([0.0, 1.0]).to(device),
                    coords + torch.tensor([-1.0, 1.0]).to(device),
                    coords + torch.tensor([-1.0, 0.0]).to(device),
                    coords + torch.tensor([-1.0, -1.0]).to(device),
                    coords + torch.tensor([0.0, -1.0]).to(device),
                    coords + torch.tensor([1.0, -1.0]).to(device)
                ],
                dim=0)
            _, surrounding_local_mask = model.local_representation(surrounding_points)
            loss_artefacts = self.weight_artefacts * self.criterion_artefacts(
                surrounding_local_mask['mask_combined'],
                torch.zeros_like(surrounding_local_mask['mask_combined'])
                )
        else:
            loss_artefacts = torch.zeros_like(loss_initital_pos)

        self.loss_for_backward += loss_initital_pos + loss_artefacts + loss_segmentation
        self.losses_epoch['loss_initital_pos'] += loss_initital_pos.detach().cpu()
        self.losses_epoch['loss_artefacts'] += loss_artefacts.detach().cpu()
        self.losses_epoch['loss_full'] += self.loss_for_backward.detach().cpu()
        self.losses_epoch['loss_segmentation'] += loss_segmentation.detach().cpu()

    def backward(self):
        self.loss_for_backward.backward()

    def zero_losses(self):
        for k in self.losses_epoch:
            self.losses_epoch[k] = 0
