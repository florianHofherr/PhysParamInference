import os
import torch

import hydra
from _learningPipeline.copyCode import copyFiles
import logging

from omegaconf import DictConfig
from models.sceneRepresentation import Scene
from dataset.dataset import ImageDataset_paig, Dataloader
from optimization.optimizers import optimizersScene
from optimization.loss import Losses
from util.visualization import VisualizationSpring
from util.initialValues import estimate_inital_vals_spring
from util.util import setSeeds


log = logging.getLogger(__name__)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="2ObjectsSpring")
def main(cfg: DictConfig):
    if 'copy' in cfg:
        copyFiles(cfg['copy'])
        log.info("Code copied")

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Seed
    setSeeds(cfg.seed)
    train_data = ImageDataset_paig(
        **cfg.data
    )

    test_data = ImageDataset_paig(
        **cfg.data,
        test_set=True
    )

    train_dataloader = Dataloader(train_data, **cfg.dataloader)

    tspan = train_data.t_steps.to(device)
    tspan_eval = test_data.t_steps.to(device)
    if cfg.loss.use_center_bb:
        traj_center1 = train_data.trajectory_bb_center1.to(device)
        traj_center2 = train_data.trajectory_bb_center2.to(device)
    else:
        traj_center1 = train_data.trajectory_mean1.to(device)
        traj_center2 = train_data.trajectory_mean2.to(device)

    # Initialize model
    model = Scene(**cfg.scene.background)

    init_values_estimate = estimate_inital_vals_spring(train_data.get_full_mask(0), train_data.get_pixel_coords())

    # Seed again to ensure consistent initialization
    # (different architectures before will change the seed at this point)
    setSeeds(cfg.seed)
    model.add_2ObjectsSpring(
        **cfg.scene.local_representation,
        **cfg.ode,
        k=torch.tensor(cfg.ode.k_init),
        **init_values_estimate
    )

    # Move to device
    model.to(device)

    # Initialize the optimizers
    optimizers = optimizersScene(model, cfg.optimizer)

    if cfg.logging.enable:
        visualization = VisualizationSpring()

    model.train()

    # Initialize loss
    losses = Losses(cfg.loss, **cfg.loss)
    losses.add_losses_spring(cfg.loss, traj_center1, traj_center2)

    # Seed again to ensure consistent initialization
    # (different architectures before will change the seed at this point)
    setSeeds(cfg.seed)

    # Run trainingsloop
    for epoch in range(0, cfg.optimizer.epochs):
        losses.zero_losses()

        for data in train_dataloader:
            # Read data
            coords = data["coords"].to(device)
            colors_gt = data["im_vals"].to(device)
            mask1_gt = data["mask1"].to(device)
            mask2_gt = data["mask2"].to(device)

            # Adjust number of frames used (online training)
            if cfg.online_training.enable:
                l_tInterval = min(epoch // cfg.online_training.stepsize + cfg.online_training.start_length, len(tspan))
                tspan_cur = tspan[:l_tInterval]
                colors_gt = colors_gt[:, :l_tInterval, :]
                mask_gt = {
                    'mask1': mask1_gt[:, :l_tInterval],
                    'mask2': mask2_gt[:, :l_tInterval]
                }
            else:
                mask_gt = {
                    'mask1': mask1_gt,
                    'mask2': mask2_gt
                }
                l_tInterval = None
                tspan_cur = tspan

            if epoch % cfg.loss.reduce_loss_segmentation_after == 0 and epoch > 0:
                losses.weight_segmentation *= cfg.loss.factor_reduction

            # Zero gradient
            optimizers.zero_grad()

            model.update_trafo(tspan_cur)

            # Do trainings step
            output = model(coords)

            # Compute losses and backpropagate
            losses.compute_losses(output, colors_gt)
            losses.compute_losses_spring(model, output, mask_gt, coords, tspan_cur, epoch)
            losses.backward()
            optimizers.optimizer_step()

            with torch.no_grad():
                model.local_representation.ode.k.clamp_(0.0)
                model.local_representation.ode.eq_distance.clamp_(0.0)

        # Adjust learning rates
        optimizers.lr_scheduler_step()

        # Write to tensorboard
        if epoch % cfg.logging.logging_interval == 0 and cfg.logging.enable:
            visualization.log_scalars(epoch, losses, model, l_tInterval)
            log.info("Scalars logged. Epoch: " + str(epoch))

        # Render test frames
        if epoch % cfg.logging.test_interval == 0 and cfg.logging.enable:
            model.eval()
            visualization.render_test(epoch, model, tspan_eval, test_data, train_data)
            model.train()
            log.info("Rendering test done. Epoch: " + str(epoch))

        # Checkpoint
        if (
            epoch % cfg.logging.checkpoint_interval == 0
            and epoch > 0
        ):
            log.info("Storing checkpoint. Epoch: " + str(epoch))
            torch.save(model.state_dict(), os.path.join(os.path.abspath(''), 'ckpt.pth'))

    log.info("Storing final checkpoint. Epoch: " + str(epoch))
    torch.save(model.state_dict(), os.path.join(os.path.abspath(''), 'ckpt.pth'))


if __name__ == "__main__":
    main()
