import torch
from skimage import morphology, measure


def get_pendulum_init(im_gt, im_rendered, thres=0.8):
    """
    Both images must be of shape (H, W, C) and be normalized
    """
    H, W = im_gt.shape[:-1]

    im_diff = torch.norm(im_rendered - im_gt, dim=-1)
    bin_im_diff = im_diff > thres
    bin_im_diff = morphology.remove_small_objects(bin_im_diff.numpy(), min_size=500)

    # Check if we have isolated one connected component. Only warning if not for now
    _, num_connected = measure.label(bin_im_diff, return_num=True)
    if num_connected > 1:
        print("More than one connected component detected! Number: ", num_connected)

    # Create coordinates, normalized to the widht
    xx = torch.arange(W, dtype=torch.float32) / W
    yy = torch.arange(H, dtype=torch.float32) / W
    x_grid, y_grid = torch.meshgrid(xx, yy)
    pixel_coords = torch.stack([x_grid.transpose(0, 1), y_grid.transpose(0, 1)], dim=2)

    # Select points that are in the difference mask
    points = pixel_coords[bin_im_diff, :]

    # Compute the mean of the points of the mask as origin of the local coordinate system
    center = torch.mean(points, dim=0)

    # Compute the principal components
    _, _, pc = torch.pca_lowrank(points)

    # Longer component needs to point down
    if pc[0, 1] < 0:
        pc *= -1

    # Extract the highest and lowest point
    ind_low = torch.min(points[:, 1], dim=0).indices
    ind_high = torch.max(points[:, 1], dim=0).indices
    diff = points[ind_low, :] - points[ind_high, :]

    # Compute the lengths of the axes of the local NeRF by projecting onto the principal components
    l_long = torch.abs(torch.dot(diff, pc[0, :]))
    l_short = torch.abs(torch.dot(diff, pc[1, :]))

    # Compute angle (between y pointing down and pigger principal component)
    phi_0 = torch.acos(pc[0, 1] / torch.norm(pc[0, :]))

    init = {
        "A": center - l_long/2 * pc[0, :],
        "t": l_long/2,
        "l_pendulum": l_long,
        "box_dim": torch.tensor([l_short/2, l_long/2], dtype=torch.float32),
        "x0": torch.tensor([phi_0, 0], dtype=torch.float32)
    }

    return init


def estimate_initial_values(masks, pixel_coords, time_steps):
    # Masks need to be in image format (Time, H, W)

    # Estimate the rotation center
    average_mask = torch.mean(masks.float(), dim=0)
    A = pixel_coords[torch.argmax(average_mask), :]

    second_idx = 1

    # Estimate box dims
    points1 = pixel_coords[masks[0].view(-1) > 0.05]
    points2 = pixel_coords[masks[second_idx].view(-1) > 0.05]

    _, _, pc1 = torch.pca_lowrank(points1)
    _, _, pc2 = torch.pca_lowrank(points2)

    # Check if directions correctly oriented
    projection_to_first_pc1 = torch.matmul(points1 - A, pc1[:, 0])

    if torch.mean(projection_to_first_pc1) < 0:
        pc1 *= -1

    projection_to_first_pc2 = torch.matmul(points2 - A, pc2[:, 0])

    if torch.mean(projection_to_first_pc2) < 0:
        pc2 *= -1

    # Estimate initial angle
    # The pixel coordinates are left handed! Therefore the "-"
    angle1 = -torch.atan2(-pc1[0, 0], pc1[1, 0])
    angle2 = -torch.atan2(-pc2[0, 0], pc2[1, 0])
    x0_1 = (angle2 - angle1) / (time_steps[second_idx] - time_steps[0])

    x0 = torch.tensor([angle1, x0_1])

    return {
        "A": A,
        "x0": x0
    }


def estimate_inital_vals_spring(masks, coords):
    p1_0 = torch.mean(coords[masks["masks1"].flatten() > 0], dim=0)
    p2_0 = torch.mean(coords[masks["masks2"].flatten() > 0], dim=0)

    return {
        "p1_0": p1_0,
        "p2_0": p2_0,
        "eq_distance": torch.norm(p1_0 - p2_0)
    }


def estimate_initial_vals_sliding_block(masks, coords):
    center1 = torch.mean(coords[masks[0].flatten() > 0], dim=0)
    center2 = torch.mean(coords[masks[1].flatten() > 0], dim=0)
    diff = center2 - center1

    # The magnitude of alpha is the angle between the difference vector and the horizontal direction
    # If the block moves to the right, the angle is positive, else it is negative (We assume an image
    # cosy with x right, y down, therefore positive rotation is clockwise)
    alpha = torch.sign(diff[0]) * torch.acos(torch.abs(diff[0]) / torch.norm(diff))

    return {
        "alpha": alpha,
        "p0": center1
    }


def estimate_initial_vals_ball(masks, coords, tspan):
    center1 = torch.mean(coords[masks[0].flatten() > 0], dim=0)
    center2 = torch.mean(coords[masks[1].flatten() > 0], dim=0)
    v0 = (center2 - center1) / (tspan[1] - tspan[0])

    return {
        "p0": center1,
        "v0": v0
    }
