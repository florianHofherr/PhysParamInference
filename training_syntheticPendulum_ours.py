import os
import torch

import hydra
import logging

from omegaconf import DictConfig
from models.sceneRepresentation import Scene
from dataset.dataset import get_split_dynamic_pixel_data, Dataloader
from util.initialValues import estimate_initial_values
from util.visualization import VisualizationSyntheticPendulum
from optimization.optimizers import optimizersScene
from optimization.loss import Losses
from util.util import setSeeds


log = logging.getLogger(__name__)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="syntheticPendulum_ours")
def main(cfg: DictConfig):

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Seed
    setSeeds(cfg.seed)

    # Load the datasets
    train_data, test_data = get_split_dynamic_pixel_data(**cfg.data)

    train_dataloader = Dataloader(train_data, **cfg.dataloader)

    tspan = train_data.parameters["tspan"].to(device)
    tspan_eval = test_data.parameters["tspan"].to(device)
    log.info("Done loading data")

    # Initialize model
    model = Scene(**cfg.scene.background)

    # Initialization parameters
    if cfg.ode.init_with_gt:
        c_init = train_data.parameters['c']
        l_pendulum_init = train_data.parameters['l_pendulum']
    else:
        c_init = torch.tensor([cfg.ode.c_init], dtype=torch.float32)
        l_pendulum_init = torch.tensor([cfg.ode.l_pendulum_init], dtype=torch.float32)

    init_values_estimate = estimate_initial_values(
        train_data.get_full_mask(),
        train_data.get_pixel_coords(),
        tspan,
    )

    rel_error_init_A = (torch.norm(train_data.get_image_dim()[0]*init_values_estimate['A'] - train_data.parameters['A']) /
                        torch.norm(train_data.parameters['A']))
    rel_error_init_x0 = torch.norm(init_values_estimate['x0'] - train_data.parameters['x0']) / torch.norm(train_data.parameters['x0'])
    log.info("Rel error init A: " + str(rel_error_init_A))
    log.info("Rel error init x0: " + str(rel_error_init_x0))

    # Seed again to ensure consistent initialization
    # (different architectures before will change the seed at this point)
    setSeeds(cfg.seed)
    model.add_pendulum(
        c=c_init,
        l_pendulum=l_pendulum_init,
        **cfg.ode,
        **init_values_estimate,
        **cfg.scene.local_representation
    )

    # Move to device
    model.to(device)

    # Initialize the optimizers
    optimizers = optimizersScene(model, cfg.optimizer)

    if cfg.logging.enable:
        visualization = VisualizationSyntheticPendulum(train_data.parameters)

    # Initialize loss
    losses = Losses(cfg.loss, **cfg.loss)
    losses.regularize_mask = False

    # Log initialization
    visualization.log_scalars(-1, losses, model, len(tspan), train_data)
    visualization.render_test(-1, model, tspan, tspan_eval, train_data, test_data)
    log.info("Initialization logged")

    # Seed again to ensure consistent initialization
    # (different architectures before will change the seed at this point)
    setSeeds(cfg.seed)

    # Run trainingsloop
    log.info("Start Trainings Loop")
    for epoch in range(cfg.optimizer.epochs):
        losses.zero_losses()

        if epoch == cfg['loss']['regularize_after_epochs']:
            losses.regularize_mask = True

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
                l_tInterval = None
                tspan_cur = tspan

            # Zero gradient
            optimizers.zero_grad()

            model.update_trafo(tspan_cur)

            # Do trainings step
            output = model(coords)

            losses.compute_losses(output, colors_gt)
            losses.backward()
            optimizers.optimizer_step()

            if cfg.ode.use_damping:
                with torch.no_grad():
                    model.local_representation.ode.c.clamp_(0.0)

        # Adjust learning rates
        optimizers.lr_scheduler_step()

        # Write scalars to tensorboard
        if epoch % cfg.logging.logging_interval == 0 and cfg.logging.enable:
            visualization.log_scalars(epoch, losses, model, l_tInterval, train_data)
            log.info("Scalars logged. Epoch: " + str(epoch))

        # Render test frames
        if epoch % cfg.logging.test_interval == 0 and cfg.logging.enable:
            visualization.render_test(epoch, model, tspan_cur, tspan_eval, train_data, test_data)
            log.info("Rendering test done. Epoch: " + str(epoch))

        # Checkpoint
        if (epoch % cfg.logging.checkpoint_interval == 0 and epoch > 0):
            log.info("Storing checkpoint. Epoch " + str(epoch))
            torch.save(model.state_dict(), os.path.join(os.path.abspath(''), 'ckpt.pth'))

    log.info("Storing final checkpoint. Epoch" + str(epoch))
    torch.save(model.state_dict(), os.path.join(os.path.abspath(''), 'ckpt.pth'))


if __name__ == "__main__":
    main()
