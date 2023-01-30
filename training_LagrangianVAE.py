import os
import torch

import hydra
import logging

from omegaconf import DictConfig
from models.sceneRepresentation import Scene
from dataset.dataset import ImageDataset_LagrangianVAE, Dataloader
from optimization.optimizers import optimizersScene
from util.util import setSeeds
from util.initialValues import estimate_initial_values
from util.visualization import VisualizationLagrangianVAE


log = logging.getLogger(__name__)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="LagrangianVAE")
def main(cfg: DictConfig):

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Seed
    setSeeds(cfg.seed)

    # Load the dataset
    dataset = ImageDataset_LagrangianVAE(**cfg.data)

    train_dataloader = Dataloader(dataset, **cfg.dataloader)

    tspan = dataset.t_steps_train.to(device)
    tspan_eval = dataset.t_steps_eval.to(device)
    log.info("Done loading data")

    # Initialize model
    model = Scene()

    # Initialization parameters
    c_init = torch.tensor([cfg.ode.c_init], dtype=torch.float32)
    l_pendulum_init = torch.tensor([cfg.ode.l_pendulum_init], dtype=torch.float32)

    init_params = estimate_initial_values(dataset.get_full_mask(), dataset.get_pixel_coords(), tspan)

    # Seed again to ensure consistent initialization
    # (different architectures before will change the seed at this point)
    setSeeds(cfg.seed)
    model.add_pendulum(
        c=c_init,
        l_pendulum=l_pendulum_init,
        **cfg.scene.local_representation,
        **cfg.ode,
        **init_params
    )

    # Move to device
    model.to(device)

    # Initialize the optimizers
    optimizers = optimizersScene(model, cfg.optimizer)

    if cfg.logging.enable:
        visualization = VisualizationLagrangianVAE()

    # Log initialization
    visualization.render_test(-1, model, dataset, tspan_eval)

    # Initialize loss criterion
    criterion = torch.nn.BCELoss()

    # Seed again to ensure consistent initialization
    # (different architectures before will change the seed at this point)
    setSeeds(cfg.seed)

    # Run trainingsloop
    for epoch in range(0, cfg.optimizer.epochs):
        loss_epoch = 0

        for data in train_dataloader:
            # Read data
            coords = data["coords"].to(device)
            mask = data["mask"].to(device)

            # Adjust number of frames used (online training)
            if cfg.online_training.enable:
                l_tInterval = min(epoch // cfg.online_training.stepsize + cfg.online_training.start_length, len(tspan))
                tspan_cur = tspan[:l_tInterval]
                mask = mask[:, :l_tInterval]
            else:
                tspan_cur = tspan
                l_tInterval = len(tspan)

            # Zero gradient
            optimizers.zero_grad()

            model.update_trafo(tspan_cur)

            # Do trainings step
            output = model(coords)

            # Loss on segmentation mask
            loss = criterion(output['mask'], mask.float())
            loss.backward()
            optimizers.optimizer_step()

            # Project to constrained set (in particular the damping)
            if cfg.ode.use_damping:
                with torch.no_grad():
                    model.local_representation.ode.c.clamp_(0.0)

            # Accumulate Loss
            loss_epoch += loss.detach().cpu()

        # Adjust learning rates
        optimizers.lr_scheduler_step()

        # Write to tensorboard
        if epoch % cfg.logging.logging_interval == 0 and cfg.logging.enable:
            visualization.log_scalars(epoch, loss_epoch, model, l_tInterval)
            log.info("Scalars logged. Epoch: " + str(epoch))

        # Render test frames
        if epoch % cfg.logging.test_interval == 0 and cfg.logging.enable:
            model.eval()
            visualization.render_test(epoch, model, dataset, tspan_eval)
            model.train()
            log.info("Rendering test done. Epoch: " + str(epoch))

        # Checkpoint
        if (epoch % cfg.logging.checkpoint_interval == 0 and epoch > 0):
            log.info("Storing checkpoint. Epoch " + str(epoch))
            torch.save(model.state_dict(), os.path.join(os.path.abspath(''), 'ckpt.pth'))

    log.info("Storing final checkpoint. Epoch" + str(epoch))
    torch.save(model.state_dict(), os.path.join(os.path.abspath(''), 'ckpt.pth'))


if __name__ == "__main__":
    main()
