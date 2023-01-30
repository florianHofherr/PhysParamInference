import os
import torch

import hydra
import logging

from omegaconf import DictConfig
from models.sceneRepresentation import Scene
from dataset.dataset import ImageDataset_realData, Dataloader
from optimization.optimizers import optimizersScene
from optimization.loss import Losses
from util.initialValues import estimate_initial_vals_sliding_block
from util.visualization import VisualizationRealSlidingBlock
from util.util import setSeeds


log = logging.getLogger(__name__)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="realSlidingBlock")
def main(cfg: DictConfig):

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Seed
    setSeeds(cfg.seed)

    train_data = ImageDataset_realData(
        **cfg.data,
        max_samples=cfg['data']['samples_train']
    )

    test_data = ImageDataset_realData(
        **cfg.data,
        indices=train_data.unused_inds
    )

    train_dataloader = Dataloader(train_data, **cfg.dataloader)

    tspan = train_data.t_steps.to(device)
    tspan_test = test_data.t_steps.to(device)
    log.info("Done loading data")

    # Initialize model
    model = Scene(**cfg.scene.background)

    # Initialization parameters
    mu_init = torch.tensor([cfg.ode.mu_init], dtype=torch.float32)

    init_values_estimate = estimate_initial_vals_sliding_block(
        train_data.get_full_mask(),
        train_data.get_pixel_coords()
    )

    # Seed again to ensure consistent initialization
    # (different architectures before will change the seed at this point)
    setSeeds(cfg.seed)
    model.add_slidingBlock(
        mu=mu_init,
        **init_values_estimate,
        **cfg.scene.local_representation
    )

    # Move to device
    model.to(device)

    # Initialize the optimizers
    optimizers = optimizersScene(model, cfg.optimizer)

    if cfg.logging.enable:
        visualization = VisualizationRealSlidingBlock()

    model.train()

    # Initialize loss
    losses = Losses(cfg.loss)
    losses.regularize_mask = False

    # Seed again to ensure consistent initialization
    # (different architectures before will change the seed at this point)
    setSeeds(cfg.seed)

    # Run trainingsloop
    log.info("Start Trainings Loop")
    for epoch in range(0, cfg.optimizer.epochs):
        losses.zero_losses()

        if epoch == cfg['loss']['regularize_after_epochs']:
            losses.regularize_mask = True

        if cfg.homography.enable and epoch == cfg.homography.enable_after_epochs:
            model.use_homography = True

        for data in train_dataloader:
            # Read data
            coords = data["coords"].to(device)
            colors_gt = data["im_vals"].to(device)

            # Adjust number of frames used (online training)
            if cfg.online_training.enable:
                l_tInterval = min(epoch // cfg.online_training.stepsize + cfg.online_training.start_length, len(tspan))
                tspan_cur = tspan[:l_tInterval]
                colors_gt = colors_gt[:, :l_tInterval, :]
            else:
                tspan_cur = tspan
                l_tInterval = len(tspan)

            # Zero gradient
            optimizers.zero_grad()

            model.update_trafo(tspan_cur)

            # Do trainings step
            output = model(coords)

            losses.compute_losses(output, colors_gt)
            losses.backward()
            optimizers.optimizer_step()

        # Adjust learning rates
        optimizers.lr_scheduler_step()

        # Write to tensorboard
        if epoch % cfg.logging.logging_interval == 0 and cfg.logging.enable:
            visualization.log_scalars(epoch, losses, model, l_tInterval, train_data)
            log.info("Scalars logged. Epoch: " + str(epoch))

        # Render test frames
        if epoch % cfg.logging.test_interval == 0 and cfg.logging.enable:
            model.eval()
            visualization.render_test(epoch, model, tspan, train_data, tspan_test, test_data)
            model.train()
            log.info("Rendering test done. Epoch: " + str(epoch))

        # Checkpoint
        if (
            epoch % cfg.logging.checkpoint_interval == 0
            and epoch > 0
        ):
            log.info("Storing checkpoint. Epoch: " + str(epoch))
            torch.save(model.state_dict(), os.path.join(os.path.abspath(''), 'ckpt.pth'))

    log.info("Storing final checkpoint. Epoch" + str(epoch))
    torch.save(model.state_dict(), os.path.join(os.path.abspath(''), 'ckpt.pth'))


if __name__ == "__main__":
    main()
