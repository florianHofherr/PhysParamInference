import torch
import random
import numpy as np


def setSeeds(seed):
    # Seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rotmat_2d(phi):
    # Case rotmat only for one angle
    if len(phi.shape) == 0:
        phi = phi.unsqueeze(0)

    # phi: 1d tensor with rotation angles in radians. Must be one dimensional
    cos_phi = torch.cos(phi).unsqueeze(1).unsqueeze(2)
    sin_phi = torch.sin(phi).unsqueeze(1).unsqueeze(2)

    R = torch.cat([cos_phi, sin_phi, -sin_phi, cos_phi], dim=1).view(-1, 2, 2).squeeze()
    return R


def applyHomography(H, x):
    result = x @ H[:2, :] + H[2, :]
    return result[:, :2] / result[:, 2].unsqueeze(1)


def compute_psnr(x, y):
    # x and y must be data with max value 1.0
    mse = torch.mean((x - y)**2)
    psnr = -10.0 * torch.log10(mse)
    return psnr


def compute_iou(x, y, threshold=0.5):
    # Make logical (does not hurt if it already is)
    x = x > threshold
    y = y > threshold

    intersection = torch.sum(torch.logical_and(x, y))
    union = torch.sum(torch.logical_or(x, y))
    return intersection / union


def get_pixel_coords(H, W, normalize_by_H=False, device="cpu"):
    # Returns pixel coords for an image with widht W and height H. Pixels normalized by the widht
    if normalize_by_H:
        normalization = H
    else:
        normalization = W
    xx = torch.arange(W, dtype=torch.float32, device=device) / normalization
    yy = torch.arange(H, dtype=torch.float32, device=device) / normalization
    y_grid, x_grid = torch.meshgrid(yy, xx)
    coords = torch.stack([x_grid, y_grid], dim=2).view(-1, 2)

    return coords


def interp1D(x, xp, fp):
    """
    Interpolate the function defined at 1D locations xp by the values fp at the evaluation points x
    Shape:
        xp: [nPoints]
        fp: [dim, nPoints]
        x:  [nEvalPoints]

        Extrapolation is done by keeping the last value
    """
    # Extrapolation: Append same value very far away
    xp = torch.cat([
        torch.tensor([-10000.0], device=xp.device),
        xp,
        torch.tensor([10000.0], device=xp.device)
    ])
    fp = torch.cat([
        fp[:, :1],
        fp,
        fp[:, -1:]
    ], dim=-1)

    diff = xp.unsqueeze(0) - x.unsqueeze(1)
    inds_2 = len(xp) - torch.sum(diff > 0, dim=1)

    f_1 = fp[..., inds_2 - 1]
    f_2 = fp[..., inds_2]

    # Compute differences to the left and right point in xp
    diff_x1 = -diff[torch.arange(len(x)), inds_2 - 1]
    diff_x2 = diff[torch.arange(len(x)), inds_2]

    return (f_2 * diff_x1 + f_1 * diff_x2) / (xp[inds_2] - xp[inds_2-1])


def get_bounding_box(mask):
    # Mask shape: H, W
    H, W = mask.shape
    pixel_coords = get_pixel_coords(H, W)

    points = pixel_coords[mask.bool().flatten()]

    xmin = torch.min(points[:, 0])
    xmax = torch.max(points[:, 0])
    ymin = torch.min(points[:, 1])
    ymax = torch.max(points[:, 1])

    bounding_box = torch.stack([xmin, xmax, ymin, ymax])
    center = torch.stack([xmin + xmax, ymin + ymax]) / 2

    return bounding_box, center


def blend_masks(predicted_mask, true_mask, factor=0.4):
    # Expects gray value masks of shape (B, H, W)
    predicted_mask = predicted_mask.float().unsqueeze(1).repeat(1, 3, 1, 1).detach().cpu().clamp(0.0, 1.0)
    true_mask = torch.stack([torch.zeros_like(true_mask), true_mask, torch.zeros_like(true_mask)], dim=1)

    blended = predicted_mask + factor*true_mask
    blended /= torch.max(blended)

    return blended
