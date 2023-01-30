import torch
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from models.sceneRepresentation import Scene
from optimization.loss import Losses
from dataset.dataset import DynamicPixelDataset, ImageDataset_realData, ImageDataset_paig, ImageDataset_LagrangianVAE
from util.util import blend_masks, compute_psnr, compute_iou
import matplotlib.pyplot as plt


class VisualizationSyntheticPendulum():
    def __init__(self, gt_parameters: DictConfig):
        self.tensorBoard = SummaryWriter()

        self.data_A = {
            'A_x_gt': gt_parameters["A"][0],
            'A_y_gt': gt_parameters["A"][1]
        }

        self.data_x0 = {
            'phi_0_gt': gt_parameters["x0"][0],
            'omega_0_gt': gt_parameters["x0"][1]
        }

        self.data_l_pendulum = {
            'l_gt': gt_parameters["l_pendulum"]
        }

        self.data_c = {
            'c_gt': gt_parameters["c"]
        }

    def log_scalars(
        self,
        epoch: int,
        losses: Losses,
        model: Scene,
        l_tInterval: int,
        train_data: DynamicPixelDataset
    ):
        _, W = train_data.get_image_dim()
        self.tensorBoard.add_scalars('Loss', losses.losses_epoch, epoch)

        self.data_A['A_x'] = W*model.local_representation.A.data[0]
        self.data_A['A_y'] = W*model.local_representation.A.data[1]
        self.tensorBoard.add_scalars('A', self.data_A, epoch)

        self.data_x0['phi_0'] = model.local_representation.x0.data[0]
        self.data_x0['omega_0'] = model.local_representation.x0.data[1]
        self.tensorBoard.add_scalars('Init values ODE', self.data_x0, epoch)

        self.data_l_pendulum['l'] = model.local_representation.ode.l_pendulum.data
        self.tensorBoard.add_scalars('l_pendulum', self.data_l_pendulum, epoch)

        if model.local_representation.ode.use_damping:
            self.data_c['c'] = model.local_representation.ode.c.data
            self.tensorBoard.add_scalars('c', self.data_c, epoch)

        if l_tInterval is not None:
            self.tensorBoard.add_scalar('Length Time Interval', l_tInterval, epoch)

    def render_test(
        self,
        epoch: int,
        model: Scene,
        tspan_train: torch.Tensor,
        tspan_eval: torch.Tensor,
        train_data: DynamicPixelDataset,
        test_data: DynamicPixelDataset
    ):
        H, W = test_data.get_image_dim()

        # Evaluate Model on train sequence
        model.update_trafo(tspan_train)
        output_train = model.render_image(W, H)

        cur_train_images = train_data.get_full_images(range(len(tspan_train))).cpu()

        self.tensorBoard.add_images(
            'Masks - train gt',
            train_data.get_full_mask().unsqueeze(-1).permute(0, 3, 1, 2),
            epoch
        )

        self.tensorBoard.add_images(
            'Full images - train',
            output_train['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt - train',
            cur_train_images.permute(0, 3, 1, 2),
            epoch
        )

        psnr_train = compute_psnr(output_train['Image'].cpu(), cur_train_images)
        self.tensorBoard.add_scalar(
            'PSNR - Average over train sequence (of current length)',
            psnr_train,
            epoch
        )

        # Evaluate Model on eval sequence
        model.update_trafo(tspan_eval)
        output = model.render_image(W, H)

        # Plot images
        self.tensorBoard.add_images(
            'Full images',
            output['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt',
            test_data.get_full_images().permute(0, 3, 1, 2),
            epoch
        )

        psnr_test = compute_psnr(output['Image'].cpu(), test_data.get_full_images())
        self.tensorBoard.add_scalar(
            'PSNR - Average over test sequence',
            psnr_test,
            epoch
        )

        iou_test = compute_iou(output['Mask'].cpu(), test_data.get_full_mask())
        self.tensorBoard.add_scalar(
            'IoU - Average over test sequence',
            iou_test,
            epoch
        )

        # Plot masks
        blended_masks = blend_masks(output['Mask'], test_data.get_full_mask())
        self.tensorBoard.add_images(
            'Rendered Masks and GT (green)',
            blended_masks,
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Masks',
            output['Mask'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Absolute Difference Masks GT - Rendered',
            torch.abs(test_data.get_full_mask() - output['Mask'].detach().cpu()).unsqueeze(1),
            epoch
        )

        # Plot Background
        output = model.render_image(W, H, only_background=True)
        self.tensorBoard.add_image(
            'Background',
            output['Image'].permute(2, 0, 1).detach().cpu(),
            epoch
        )

        # Plot local
        local_mask, local_colors = model.local_representation.render_local_object(200, 420)
        self.tensorBoard.add_image(
            'Local Mask',
            local_mask.unsqueeze(0).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_image(
            'Local Colors',
            local_colors.permute(2, 0, 1).detach().cpu(),
            epoch
        )


class VisualizationRealPendulum():
    def __init__(self):
        self.tensorBoard = SummaryWriter()

    def log_scalars(
        self,
        epoch: int,
        losses: Losses,
        model: Scene,
        l_tInterval: int
    ):
        self.tensorBoard.add_scalars('Loss', losses.losses_epoch, epoch)

        data_A = {
            'A_x': model.local_representation.A.data[0],
            'A_y': model.local_representation.A.data[1]
        }
        self.tensorBoard.add_scalars('A', data_A, epoch)

        data_x0 = {
            'phi_0': model.local_representation.x0.data[0],
            'omega_0': model.local_representation.x0.data[1]
        }
        self.tensorBoard.add_scalars('Init values ODE', data_x0, epoch)

        self.tensorBoard.add_scalar('l_pendulum', model.local_representation.ode.l_pendulum.data, epoch)

        if model.local_representation.ode.use_damping:
            self.tensorBoard.add_scalar('c', model.local_representation.ode.c.data, epoch)

        if l_tInterval is not None:
            self.tensorBoard.add_scalar('Length Time Interval', l_tInterval, epoch)

    def render_test(
        self,
        epoch: int,
        model: Scene,
        tspan_train: torch.Tensor,
        train_data: ImageDataset_realData,
        tspan_test: torch.Tensor,
        test_data: ImageDataset_realData
    ):
        H, W = train_data.get_image_dim()

        # Evaluate Model on train sequence
        model.update_trafo(tspan_train)
        output = model.render_image(W, H, normalize_by_H=True)

        # Plot images
        self.tensorBoard.add_images(
            'Full images - train',
            output['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt - train',
            train_data.get_full_images().cpu().permute(0, 3, 1, 2),
            epoch
        )

        psnr_train = compute_psnr(output['Image'].cpu(), train_data.get_full_images().cpu())
        self.tensorBoard.add_scalar(
            'PSNR - Average over train sequence',
            psnr_train,
            epoch
        )

        # Plot masks
        blended_masks = blend_masks(output['Mask'], train_data.get_full_mask().cpu())
        self.tensorBoard.add_images(
            'Rendered Masks and GT (green) - train',
            blended_masks,
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Masks - train',
            output['Mask'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Absolute Difference Masks GT - Rendered - train',
            torch.abs(train_data.get_full_mask().cpu() - output['Mask'].detach().cpu()).unsqueeze(1),
            epoch
        )

        # Evaluate Model on test sequence
        model.update_trafo(tspan_test)
        output = model.render_image(W, H, normalize_by_H=True)

        # Plot images
        self.tensorBoard.add_images(
            'Full images - test',
            output['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt - test',
            test_data.get_full_images().cpu().permute(0, 3, 1, 2),
            epoch
        )

        psnr_test = compute_psnr(output['Image'].cpu(), test_data.get_full_images().cpu())
        self.tensorBoard.add_scalar(
            'PSNR - Average over test sequence',
            psnr_test,
            epoch
        )

        # Plot masks
        blended_masks = blend_masks(output['Mask'], test_data.get_full_mask().cpu())
        self.tensorBoard.add_images(
            'Rendered Masks and GT (green) - test',
            blended_masks,
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Masks - test',
            output['Mask'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Absolute Difference Masks GT - Rendered - test',
            torch.abs(test_data.get_full_mask().cpu() - output['Mask'].detach().cpu()).unsqueeze(1),
            epoch
        )

        # Plot Background
        output = model.render_image(W, H, only_background=True, normalize_by_H=True)
        self.tensorBoard.add_image(
            'Background',
            output['Image'].permute(2, 0, 1).detach().cpu(),
            epoch
        )

        # Plot local
        local_mask, local_colors = model.local_representation.render_local_object(200, 420)
        self.tensorBoard.add_image(
            'Local Mask',
            local_mask.unsqueeze(0).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_image(
            'Local Colors',
            local_colors.permute(2, 0, 1).detach().cpu(),
            epoch
        )


class VisualizationRealSlidingBlock():
    def __init__(self):
        self.tensorBoard = SummaryWriter()

    def log_scalars(
        self,
        epoch: int,
        losses: Losses,
        model: Scene,
        l_tInterval: int,
        train_data: ImageDataset_realData
    ):
        self.tensorBoard.add_scalars('Loss', losses.losses_epoch, epoch)

        data_p0 = {
            'p0_x': model.local_representation.p0.data[0],
            'p0_y': model.local_representation.p0.data[1]
        }
        self.tensorBoard.add_scalars('p0', data_p0, epoch)

        data_x0 = {
            'x0': model.local_representation.x0.data[0],
            'v_0': model.local_representation.x0.data[1]
        }
        self.tensorBoard.add_scalars('Init values ODE', data_x0, epoch)

        self.tensorBoard.add_scalar('alpha', model.local_representation.ode.alpha.data, epoch)
        self.tensorBoard.add_scalar('mu', model.local_representation.ode.mu.data, epoch)

        if l_tInterval is not None:
            self.tensorBoard.add_scalar('Length Time Interval', l_tInterval, epoch)

    def render_test(
        self,
        epoch: int,
        model: Scene,
        tspan_train: torch.Tensor,
        train_data: ImageDataset_realData,
        tspan_test: torch.Tensor,
        test_data: ImageDataset_realData
    ):
        H, W = train_data.get_image_dim()

        # Evaluate Model on train sequence
        model.update_trafo(tspan_train)
        output = model.render_image(W, H)

        # Plot images
        self.tensorBoard.add_images(
            'Full images - train',
            output['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt - train',
            train_data.get_full_images().cpu().permute(0, 3, 1, 2),
            epoch
        )

        psnr_train = compute_psnr(output['Image'].cpu(), train_data.get_full_images().cpu())
        self.tensorBoard.add_scalar(
            'PSNR - Average over train sequence',
            psnr_train,
            epoch
        )

        # Plot masks
        blended_masks = blend_masks(output['Mask'], train_data.get_full_mask().cpu())
        self.tensorBoard.add_images(
            'Rendered Masks and GT (green) - train',
            blended_masks,
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Masks - train',
            output['Mask'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Absolute Difference Masks GT - Rendered - train',
            torch.abs(train_data.get_full_mask().cpu() - output['Mask'].detach().cpu()).unsqueeze(1),
            epoch
        )

        # Evaluate Model on test sequence
        model.update_trafo(tspan_test)
        output = model.render_image(W, H)

        # Plot images
        self.tensorBoard.add_images(
            'Full images - test',
            output['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt - test',
            test_data.get_full_images().cpu().permute(0, 3, 1, 2),
            epoch
        )

        psnr_test = compute_psnr(output['Image'].cpu(), test_data.get_full_images().cpu())
        self.tensorBoard.add_scalar(
            'PSNR - Average over test sequence',
            psnr_test,
            epoch
        )

        # Plot masks
        blended_masks = blend_masks(output['Mask'], test_data.get_full_mask().cpu())
        self.tensorBoard.add_images(
            'Rendered Masks and GT (green) - test',
            blended_masks,
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Masks - test',
            output['Mask'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Absolute Difference Masks GT - Rendered - test',
            torch.abs(test_data.get_full_mask().cpu() - output['Mask'].detach().cpu()).unsqueeze(1),
            epoch
        )

        # Plot Background
        output = model.render_image(W, H, only_background=True)
        self.tensorBoard.add_image(
            'Background',
            output['Image'].permute(2, 0, 1).detach().cpu(),
            epoch
        )

        # Plot local
        local_mask, local_colors = model.local_representation.render_local_object(200, 420)
        self.tensorBoard.add_image(
            'Local Mask',
            local_mask.unsqueeze(0).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_image(
            'Local Colors',
            local_colors.permute(2, 0, 1).detach().cpu(),
            epoch
        )


class VisualizationRealBall():
    def __init__(self):
        self.tensorBoard = SummaryWriter()

    def log_scalars(
        self,
        epoch: int,
        losses: Losses,
        model: Scene,
        l_tInterval: int,
        train_data: ImageDataset_realData
    ):
        self.tensorBoard.add_scalars('Loss', losses.losses_epoch, epoch)

        data_p0 = {
            'p0_x': model.local_representation.p0.data[0],
            'p0_y': model.local_representation.p0.data[1]
        }
        self.tensorBoard.add_scalars('p0', data_p0, epoch)

        data_v0 = {
            'v0_x': model.local_representation.v0.data[0],
            'v0_y': model.local_representation.v0.data[1]
        }
        self.tensorBoard.add_scalars('Initial velocities', data_v0, epoch)

        if l_tInterval is not None:
            self.tensorBoard.add_scalar('Length Time Interval', l_tInterval, epoch)

    def render_test(
        self,
        epoch: int,
        model: Scene,
        tspan_train: torch.Tensor,
        train_data: ImageDataset_realData,
        tspan_test: torch.Tensor,
        test_data: ImageDataset_realData
    ):
        H, W = train_data.get_image_dim()

        # Evaluate Model on train sequence
        model.update_trafo(tspan_train)
        output = model.render_image(W, H)

        # Plot images
        self.tensorBoard.add_images(
            'Full images - train',
            output['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt - train',
            train_data.get_full_images().cpu().permute(0, 3, 1, 2),
            epoch
        )

        psnr_train = compute_psnr(output['Image'].cpu(), train_data.get_full_images().cpu())
        self.tensorBoard.add_scalar(
            'PSNR - Average over train sequence',
            psnr_train,
            epoch
        )

        # Plot masks
        blended_masks = blend_masks(output['Mask'], train_data.get_full_mask().cpu())
        self.tensorBoard.add_images(
            'Rendered Masks and GT (green) - train',
            blended_masks,
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Masks - train',
            output['Mask'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Absolute Difference Masks GT - Rendered - train',
            torch.abs(train_data.get_full_mask().cpu() - output['Mask'].detach().cpu()).unsqueeze(1),
            epoch
        )

        # Evaluate Model on test sequence
        model.update_trafo(tspan_test)
        output = model.render_image(W, H)

        # Plot images
        self.tensorBoard.add_images(
            'Full images - test',
            output['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt - test',
            test_data.get_full_images().cpu().permute(0, 3, 1, 2),
            epoch
        )

        psnr_test = compute_psnr(output['Image'].cpu(), test_data.get_full_images().cpu())
        self.tensorBoard.add_scalar(
            'PSNR - Average over test sequence',
            psnr_test,
            epoch
        )

        # Plot masks
        blended_masks = blend_masks(output['Mask'], test_data.get_full_mask().cpu())
        self.tensorBoard.add_images(
            'Rendered Masks and GT (green) - test',
            blended_masks,
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Masks - test',
            output['Mask'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Absolute Difference Masks GT - Rendered - test',
            torch.abs(test_data.get_full_mask().cpu() - output['Mask'].detach().cpu()).unsqueeze(1),
            epoch
        )

        # Plot Background
        output = model.render_image(W, H, only_background=True)
        self.tensorBoard.add_image(
            'Background',
            output['Image'].permute(2, 0, 1).detach().cpu(),
            epoch
        )

        # Plot local
        local_mask, local_colors = model.local_representation.render_local_object(200, 420)
        self.tensorBoard.add_image(
            'Local Mask',
            local_mask.unsqueeze(0).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_image(
            'Local Colors',
            local_colors.permute(2, 0, 1).detach().cpu(),
            epoch
        )


class VisualizationLagrangianVAE():
    def __init__(self):
        self.tensorBoard = SummaryWriter()
        self.criterion_mse = torch.nn.MSELoss()

    def log_scalars(
        self,
        epoch: int,
        loss_epoch,
        model: Scene,
        l_tInterval: int
    ):
        self.tensorBoard.add_scalar('Loss', loss_epoch, epoch)

        self.tensorBoard.add_scalars(
            'A',
            {
                'A_x': model.local_representation.A.data[0],
                'A_y': model.local_representation.A.data[1],
            },
            epoch
        )

        self.tensorBoard.add_scalars(
            'x0',
            {
                'x0_0': model.local_representation.x0.data[0],
                'x0_1': model.local_representation.x0.data[1],
            },
            epoch
        )

        if model.local_representation.ode.use_damping:
            self.tensorBoard.add_scalars(
                'c',
                {
                    'c': model.local_representation.ode.c.data,
                },
                epoch
            )

        if l_tInterval is not None:
            self.tensorBoard.add_scalar('Length Interval', l_tInterval, epoch)

        self.tensorBoard.add_scalars(
            'l_pendulum',
            {
                'l': model.local_representation.ode.l_pendulum.data,
            },
            epoch
        )

    def render_test(
        self,
        epoch: int,
        model: Scene,
        data: ImageDataset_LagrangianVAE,
        tspan_eval: torch.Tensor,
    ):
        H, W = data.get_image_dim()

        # Plot masks evaluation time steps
        model.update_trafo(tspan_eval)
        output = model.render_image(W, H)
        output_mask = output["Mask"].detach().cpu()

        eval_masks = data.get_full_mask(eval_data=True)
        blended_masks = blend_masks(output_mask, eval_masks)

        self.tensorBoard.add_images("Masks blended", blended_masks, epoch)
        self.tensorBoard.add_images("Masks", output_mask.unsqueeze(-1).permute(0, 3, 1, 2).clamp(0.0, 1.0), epoch)

        iou_test = compute_iou(output_mask, eval_masks)
        self.tensorBoard.add_scalar(
            'IoU - Average over test sequence',
            iou_test,
            epoch
        )

        # Plot MSE
        mse_masks = self.criterion_mse(output_mask, eval_masks)
        output_mask[output_mask > 0.5] = 1.0
        output_mask[output_mask <= 0.5] = 0.0
        mse_masks_corrected = self.criterion_mse(output_mask, eval_masks)
        self.tensorBoard.add_scalars(
            'MSE pixels',
            {
                'MSE pixels': mse_masks,
                'MSE pixels corrected': mse_masks_corrected
            },
            epoch
        )

        # Plot Masks in high res
        output = model.render_image(1000, 1000)
        output_mask_high_res = output["Mask"].detach().cpu()
        self.tensorBoard.add_images("Masks high res", output_mask_high_res.unsqueeze(-1).permute(0, 3, 1, 2).clamp(0.0, 1.0), epoch)

        self.tensorBoard.add_scalar('Max value Mask output', torch.max(output_mask), epoch)

        # Plot local mask
        local_mask, _ = model.local_representation.render_local_object(100, 120)
        self.tensorBoard.add_image('Local Mask', local_mask.unsqueeze(0).clamp(0.0, 1.0), epoch)


class VisualizationSpring():
    def __init__(self):
        self.tensorBoard = SummaryWriter()

    def log_scalars(
        self,
        epoch: int,
        losses: Losses,
        model: Scene,
        l_tInterval: int
    ):
        self.tensorBoard.add_scalars('Loss', losses.losses_epoch, epoch)

        data_x0 = {
            'x0_1': model.local_representation.x0.data[0],
            'y0_1': model.local_representation.x0.data[1],
            'x0_2': model.local_representation.x0.data[2],
            'y0_2': model.local_representation.x0.data[3],
        }
        self.tensorBoard.add_scalars('Initial positions', data_x0, epoch)

        self.tensorBoard.add_scalar(
            'Extimated lengt l*W',
            model.local_representation.ode.eq_distance.data*64,
            epoch
        )

        if l_tInterval is not None:
            self.tensorBoard.add_scalar('Length Time Interval', l_tInterval, epoch)

        self.tensorBoard.add_scalar('k', model.local_representation.ode.k.data, epoch)
        self.tensorBoard.add_scalar('l', model.local_representation.ode.eq_distance.data, epoch)

    def render_test(
        self,
        epoch: int,
        model: Scene,
        tspan_eval: torch.Tensor,
        test_data: ImageDataset_paig,
        train_data: ImageDataset_paig
    ):
        device = next(model.parameters()).device
        H, W = test_data.get_image_dim()

        # Evaluate Model on eval sequence
        model.update_trafo(tspan_eval)
        output = model.render_image(W, H)

        # Plot images
        self.tensorBoard.add_images(
            'Full images',
            output['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt',
            test_data.get_full_images().permute(0, 3, 1, 2),
            epoch
        )

        # Plot masks
        blended_masks = blend_masks(output['Mask'], test_data.get_full_mask(combined=True))
        self.tensorBoard.add_images(
            'Rendered Masks and GT (green)',
            blended_masks,
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Masks',
            output['Mask'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Mask 1',
            output['Mask1'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Mask 2',
            output['Mask2'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )

        self.tensorBoard.add_images(
            'Absolute Difference Masks GT - Rendered',
            torch.abs(test_data.get_full_mask(combined=True).float() - output['Mask'].detach().cpu()).unsqueeze(1),
            epoch
        )

        train_masks = train_data.get_full_mask()
        self.tensorBoard.add_images(
            'Train Masks 1',
            train_masks['masks1'].unsqueeze(-1).permute(0, 3, 1, 2),
            epoch
        )
        self.tensorBoard.add_images(
            'Train Masks 2',
            train_masks['masks2'].unsqueeze(-1).permute(0, 3, 1, 2),
            epoch
        )

        fig = plt.figure()
        origin = torch.zeros(1, 2).to(device)
        global_origin1 = W*model.local_representation.trafo_from_local1(origin).squeeze().detach().cpu()
        global_origin2 = W*model.local_representation.trafo_from_local2(origin).squeeze().detach().cpu()
        for n_fig in range(blended_masks.shape[0]):
            sub = fig.add_subplot(3, 10, n_fig+1)
            sub.imshow(blended_masks[n_fig].permute(1, 2, 0))
            sub.plot(global_origin1[n_fig, 0], global_origin1[n_fig, 1], 'rx')
            sub.plot(global_origin2[n_fig, 0], global_origin2[n_fig, 1], 'bx')
            sub.plot(
                [global_origin1[n_fig, 0], global_origin2[n_fig, 0]],
                [global_origin1[n_fig, 1], global_origin2[n_fig, 1]], 'r--'
            )
            sub.plot(
                global_origin1[n_fig, 0],
                global_origin1[n_fig, 1], 'r:'
            )
            sub.plot(
                global_origin2[n_fig, 0],
                global_origin2[n_fig, 1], 'r:'
            )
        plt.show()
        self.tensorBoard.add_figure('Masks Blended And Origins', fig, epoch)

        # Plot Background
        output = model.render_image(W, H, only_background=True)
        self.tensorBoard.add_image(
            'Background',
            output['Image'].permute(2, 0, 1).detach().cpu(),
            epoch
        )

        # Plot local
        output_local = model.local_representation.render_local_object(400, 400)
        self.tensorBoard.add_image(
            'Local Mask 1',
            output_local['mask1'].unsqueeze(0).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_image(
            'Local Mask 2',
            output_local['mask2'].unsqueeze(0).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_image(
            'Local Colors 1',
            output_local['colors1'].permute(2, 0, 1).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_image(
            'Local Colors 2',
            output_local['colors2'].permute(2, 0, 1).detach().cpu(),
            epoch
        )


class VisualizationPosedLocal():
    def __init__(self, gt_parameters: DictConfig):
        self.tensorBoard = SummaryWriter()

        self.data_A = {
            'A_x_gt': gt_parameters["A"][0],
            'A_y_gt': gt_parameters["A"][1]
        }

    def log_scalars(
        self,
        epoch: int,
        losses: Losses,
        model: Scene,
        l_tInterval: int,
        train_data: DynamicPixelDataset
    ):
        self.tensorBoard.add_scalars('Loss', losses.losses_epoch, epoch)

        if l_tInterval is not None:
            self.tensorBoard.add_scalar('Length Time Interval', l_tInterval, epoch)

    def render_test(
        self,
        epoch: int,
        model: Scene,
        tspan_train: torch.Tensor,
        tspan_eval: torch.Tensor,
        train_data: DynamicPixelDataset,
        test_data: DynamicPixelDataset
    ):
        H, W = test_data.get_image_dim()

        # Evaluate Model on train sequence
        model.update_trafo(tspan_train)
        output_train = model.render_image(W, H)

        cur_train_images = train_data.get_full_images(range(len(tspan_train))).cpu()

        self.tensorBoard.add_images(
            'Full images - train',
            output_train['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt - train',
            cur_train_images.permute(0, 3, 1, 2),
            epoch
        )

        psnr_train = compute_psnr(output_train['Image'].cpu(), cur_train_images)
        self.tensorBoard.add_scalar(
            'PSNR - Average over train sequence (of current length)',
            psnr_train,
            epoch
        )

        iou_test = compute_iou(output_train['Mask'].cpu(), train_data.get_full_mask(range(len(tspan_train))))
        self.tensorBoard.add_scalar(
            'IoU - Average over train sequence',
            iou_test,
            epoch
        )

        # Plot masks
        blended_masks = blend_masks(output_train['Mask'], train_data.get_full_mask(range(len(tspan_train))).cpu())
        self.tensorBoard.add_images(
            'Rendered Masks and GT (green) - train',
            blended_masks,
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Masks - train',
            output_train['Mask'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )

        # Evaluate Model on eval sequence
        model.update_trafo(tspan_eval)
        output = model.render_image(W, H)

        # Plot images
        self.tensorBoard.add_images(
            'Full images',
            output['Image'].permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Full images gt',
            test_data.get_full_images().permute(0, 3, 1, 2),
            epoch
        )

        psnr_test = compute_psnr(output['Image'].cpu(), test_data.get_full_images())
        self.tensorBoard.add_scalar(
            'PSNR - Average over test sequence',
            psnr_test,
            epoch
        )

        iou_test = compute_iou(output['Mask'].cpu(), test_data.get_full_mask())
        self.tensorBoard.add_scalar(
            'IoU - Average over test sequence',
            iou_test,
            epoch
        )

        # Plot masks
        blended_masks = blend_masks(output['Mask'], test_data.get_full_mask())
        self.tensorBoard.add_images(
            'Rendered Masks and GT (green)',
            blended_masks,
            epoch
        )
        self.tensorBoard.add_images(
            'Rendered Masks',
            output['Mask'].unsqueeze(-1).permute(0, 3, 1, 2).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_images(
            'Absolute Difference Masks GT - Rendered',
            torch.abs(test_data.get_full_mask() - output['Mask'].detach().cpu()).unsqueeze(1),
            epoch
        )

        # Plot Background
        output = model.render_image(W, H, only_background=True)
        self.tensorBoard.add_image(
            'Background',
            output['Image'].permute(2, 0, 1).detach().cpu(),
            epoch
        )

        # Plot local
        local_mask, local_colors = model.local_representation.render_local_object(200, 420)
        self.tensorBoard.add_image(
            'Local Mask',
            local_mask.unsqueeze(0).detach().cpu(),
            epoch
        )
        self.tensorBoard.add_image(
            'Local Colors',
            local_colors.permute(2, 0, 1).detach().cpu(),
            epoch
        )
