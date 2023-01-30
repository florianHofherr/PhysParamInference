import os
import torch

import hydra
import logging

from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from models.sceneRepresentation import DynamicBackground
from dataset.dataset import get_split_dynamic_pixel_data, Dataloader
from util.util import compute_psnr, setSeeds


log = logging.getLogger(__name__)
CONFIG_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "configs")


@hydra.main(config_path=CONFIG_DIR, config_name="syntheticPendulum_dynamicBackground")
def main(cfg: DictConfig):

    # Select device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # Seed
    setSeeds(cfg.seed)

    # Load the datasets
    train_data, test_data = get_split_dynamic_pixel_data(
        **cfg.data
    )

    train_dataloader = Dataloader(train_data, **cfg.dataloader)

    tspan = train_data.parameters["tspan"].to(device)
    tspan_eval = test_data.parameters["tspan"].to(device)

    # Normalize time
    if cfg.normalize_time_interval:
        time_normalization = tspan_eval[-1]
    else:
        time_normalization = torch.tensor(1.0)

    # Initialize model
    model = DynamicBackground(
        time_normalization=time_normalization,
        **cfg.scene
    )

    # Move to device
    model.to(device)

    # Initialize the optimizers
    optim_mlp = torch.optim.Adam(
        model.parameters(),
        lr=cfg.optimizer.lr_repr
    )

    # Learning rate scheduler
    # learning rate: current_lr = base_lr * gamma ** (epoch / step_size)
    def lr_lambda_mlp(epoch):
        return cfg.optimizer.lr_scheduler_gamma_repr ** (
            epoch / cfg.optimizer.lr_scheduler_step_size_repr
        )

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    lr_scheduler_mlp = torch.optim.lr_scheduler.LambdaLR(
        optim_mlp, lr_lambda_mlp, last_epoch=-1, verbose=False
    )

    if cfg.logging.enable:
        tensorBoard = SummaryWriter()

    model.train()

    # Initialize loss
    criterion = torch.nn.MSELoss()

    # Seed again to ensure consistent initialization
    # (different architectures before will change the seed at this point)
    setSeeds(cfg.seed)

    # Run trainingsloop
    for epoch in range(0, cfg.optimizer.epochs):
        loss_epoch = 0

        for data in train_dataloader:
            # Read data
            coords = data["coords"].to(device)
            im_vals = data["im_vals"].to(device)
            mask = data["mask"].to(device)

            # Adjust number of frames used (online training)
            if cfg.online_training.enable:
                ind = min(epoch // cfg.online_training.stepsize + cfg.online_training.start_length, len(tspan))
                tspan_cur = tspan[:ind]
                im_vals = im_vals[:, :ind, :]
                mask = mask[:, :ind]
            else:
                tspan_cur = tspan

            # Zero gradient
            optim_mlp.zero_grad()

            # Do trainings step
            colors = model(coords, tspan_cur)

            # Loss on color values
            loss = criterion(im_vals.reshape(-1, 3), colors.reshape(-1, 3))

            loss.backward()
            optim_mlp.step()

            # Accumulate Loss
            loss_epoch += loss.detach().cpu()

        # Adjust learning rates
        lr_scheduler_mlp.step()

        # Write to tensorboard
        if epoch % cfg.logging.logging_interval == 0 and tensorBoard is not None:
            tensorBoard.add_scalar('Loss', loss_epoch, epoch)
            log.info(f"Loss at Epoch: {epoch}: {loss_epoch}")

            if cfg.online_training.enable:
                tensorBoard.add_scalar('Length Interval', ind, epoch)
            H, W = test_data.get_image_dim()
            tensorBoard.add_scalar('Width Image', W, epoch)

        # Render test frames
        if epoch % cfg.logging.test_interval == 0 and tensorBoard is not None:
            model.eval()
            H, W = test_data.get_image_dim()

            # Evaluate Model on train sequence
            output_train = model.render_image(W, H, tspan_cur)

            cur_train_images = train_data.get_full_images(range(len(tspan_cur))).cpu()

            tensorBoard.add_images(
                'Full images - train',
                output_train['Image'].permute(0, 3, 1, 2).detach().cpu(),
                epoch
            )
            tensorBoard.add_images(
                'Full images gt - train',
                cur_train_images.permute(0, 3, 1, 2),
                epoch
            )

            psnr_train = compute_psnr(output_train['Image'].cpu(), cur_train_images)
            tensorBoard.add_scalar(
                'PSNR - Average over train sequence (of current length)',
                psnr_train,
                epoch
            )

            # Evaluate Model
            output = model.render_image(W, H, tspan_eval)
            image = output["Image"].detach().cpu()

            # Plot images
            tensorBoard.add_images("Full images", image.permute(0, 3, 1, 2).clamp(0.0, 1.0), epoch)
            tensorBoard.add_images("Full images gt", test_data.get_full_images().permute(0, 3, 1, 2), epoch)

            tensorBoard.add_scalar("PSNR full sequence", compute_psnr(image, test_data.get_full_images()), epoch)

            model.train()
            print("Rendering test done")

        # Checkpoint
        if (
            epoch % cfg.logging.checkpoint_interval == 0
            and epoch > 0
        ):
            print("Storing checkpoint.")
            torch.save(model.state_dict(), os.path.join(os.path.abspath(''), 'ckpt.pth'))


if __name__ == "__main__":
    main()
