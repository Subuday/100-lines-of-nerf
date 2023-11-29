import json
import imageio.v2 as imageio
import configargparse
import os
import cv2
import numpy as np
import torch
import logging
import torch.nn.functional as F
from tqdm import trange
from nerf import Embedder, NeRF, run_coarse_nerf, run_fine_nerf
from PIL import Image


def create_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--name", type=str, help='experiment name')
    parser.add_argument("--data_dir", type=str, default='data/lego')
    parser.add_argument("--learning_rate_decay_steps", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--num_coarse_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--num_fine_samples", type=int, default=0, help='number of fine samples per ray')
    parser.add_argument("--batch_size_random_rays", type=int, default=1024, help='number of random rays per batch')
    parser.add_argument("--use_reduced_resolution", action='store_true', help='use reduced resolution for training')

    parser.add_argument("--step_print", type=int, default=100, help='frequency of console printout and metric loggin')
    parser.add_argument("--step_ckpt", type=int, default=10000, help='frequency of weight ckpt saving')
    parser.add_argument("--step_video", type=int, default=50000, help='frequency of render_poses video saving')
    return parser


def parse_args():
    parser = create_parser()
    model_args = parser.parse_args()
    return model_args


def load_data():
    splits = ['train', 'val', 'test']

    metas = {}
    for split in splits:
        with open(os.path.join(args.data_dir, 'transforms_{}.json'.format(split)), 'r') as fp:
            metas[split] = json.load(fp)

    images = []
    camera_to_world_transformations = []
    counts = [0]
    for split in splits:
        meta = metas[split]

        slit_images = []
        for frame in meta['frames']:
            fileName = os.path.join(args.data_dir, split, os.path.basename(frame['file_path']) + '.png')
            image = np.array(Image.open(fileName))
            slit_images.append(image)

        split_transformations = []
        for frame in meta['frames']:
            split_transformations.append(np.array(frame['transform_matrix']))

        slit_images = (np.array(slit_images) / 255.).astype(np.float64)
        images.append(slit_images)

        split_transformations = np.array(split_transformations).astype(np.float64)
        camera_to_world_transformations.append(split_transformations)

        counts.append(counts[-1] + len(slit_images))

    images = np.concatenate(images, 0)
    camera_to_world_transformations = np.concatenate(camera_to_world_transformations, 0)

    h, w = images[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * w / np.tan(.5 * camera_angle_x)

    splits = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    if args.use_reduced_resolution:
        h = h // 2
        w = w // 2
        focal = focal / 2.

        reduced_images = np.zeros((image.shape[0], h, w, 4))
        for i, img in enumerate(images):
            reduced_images[i] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        images = reduced_images

    return {
        'images': images[..., :3],
        'camera_to_world_transformations': camera_to_world_transformations,
        'hwf': [int(h), int(w), focal],
        'splits': splits
    }


def create_rays(h, w, ict, c2w):
    i, j = torch.meshgrid(
        torch.linspace(0, w - 1, w),
        torch.linspace(0, h - 1, h)
    )
    i = i.t()
    j = j.t()

    normalized_pixel_directions = torch.stack(
        [(i - ict[0][2]) / ict[0][0], -(j - ict[1][2]) / ict[1][1], -torch.ones_like(i)],
        -1
    )
    rays_d = torch.sum(normalized_pixel_directions[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)

    return rays_o, rays_d


def create_nerf():
    embedder_pts = Embedder(max_encoding_resolution=10)
    embedder_dirs = Embedder(max_encoding_resolution=4)

    coarse_model = NeRF(input_ch_pts=embedder_pts.output_dim, input_ch_views=embedder_dirs.output_dim)
    fine_model = NeRF(input_ch_pts=embedder_pts.output_dim, input_ch_views=embedder_dirs.output_dim)

    return {
        'embedder_pts': embedder_pts,
        'embedder_dirs': embedder_dirs,
        'coarse_model': coarse_model,
        'fine_model': fine_model
    }


def sample_pdf(bins, weights, N_samples, det=True):
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


def raw2outputs(raw, z_vals, rays_d, raw_noise_std):
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(raw) * dists)

    # calculates the distances between these sample points along each ray,
    # essentially telling us how far apart these samples (for color and opacity) are from each other
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[..., :1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1. - alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[..., None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    return rgb_map, disp_map, acc_map, weights, depth_map


def render_rays(nerf, rays_o, rays_d, views, near, far, noise_std):
    normalized_sample_position = torch.linspace(0., 1., steps=args.num_coarse_samples)
    sample_z_coarse = near * (1. - normalized_sample_position) + far * normalized_sample_position
    rays_num = rays_d.shape[0]
    sample_z_coarse = sample_z_coarse.expand([rays_num, args.num_coarse_samples])

    sample_points_coarse = rays_o[..., None, :] + rays_d[..., None, :] * sample_z_coarse[..., :, None]
    coarse_raw = run_coarse_nerf(nerf, sample_points_coarse, views)
    rgb_map_coarse, _, _, weights_coarse, _ = raw2outputs(coarse_raw, sample_z_coarse, rays_d, noise_std)

    sample_z_mid = .5 * (sample_z_coarse[..., 1:] + sample_z_coarse[..., :-1])
    sample_z_pdf = sample_pdf(sample_z_mid, weights_coarse[..., 1:-1], args.num_fine_samples)
    sample_z_fine, _ = torch.sort(torch.cat([sample_z_coarse, sample_z_pdf], -1), -1)

    sample_points_fine = rays_o[..., None, :] + rays_d[..., None, :] * sample_z_fine[..., :, None]
    fine_raw = run_fine_nerf(nerf, sample_points_fine, views)
    rgb_map_fine, _, _, _, _ = raw2outputs(fine_raw, sample_z_fine, rays_d, noise_std)

    if torch.isnan(rgb_map_coarse).any() or torch.isinf(rgb_map_coarse).any():
        logging.warning(f"[Numerical Error] rgb_map_coarse contains nan or inf.")

    if torch.isnan(rgb_map_fine).any() or torch.isinf(rgb_map_fine).any():
        logging.warning(f"[Numerical Error] rgb_map_fine contains nan or inf.")

    return rgb_map_coarse, rgb_map_fine


def render(nerf, rays_o, rays_d, near, far, noise_std):
    ray_d_norm = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)

    near = near * torch.ones_like(rays_d[..., :1])
    far = far * torch.ones_like(rays_d[..., :1])

    return render_rays(nerf, rays_o, rays_d, ray_d_norm, near, far, noise_std)


def mse(x1, x2):
    return torch.mean((x1 - x2) ** 2)


def mse2psnr(x):
    return -10. * torch.log(x) / torch.log(torch.Tensor([10.]))


def train():
    data = load_data()
    images = data['images']
    camera_to_world_transformations = data['camera_to_world_transformations']
    h, w, focal = data['hwf']
    split_train, split_val, split_test = data['splits']

    intrinsic_camera_transformation = torch.tensor([
        [focal, 0, 0.5 * w],
        [0, focal, 0.5 * h],
        [0, 0, 1]
    ])

    nerf = create_nerf()

    params = list(nerf['coarse_model'].parameters()) + list(nerf['fine_model'].parameters())
    optimizer = torch.optim.Adam(params=params, lr=5e-4, betas=(0.9, 0.999))

    start = 1
    iters = 200_000 + 1
    for step in trange(start, iters):
        selected_index = np.random.choice(split_train)
        image = torch.tensor(images[selected_index])
        camera_to_world_transformation = torch.tensor(camera_to_world_transformations[selected_index])

        coords = torch.stack(
            torch.meshgrid(
                torch.linspace(0, h - 1, h, dtype=torch.int),
                torch.linspace(0, w - 1, w, dtype=torch.int)
            ),
            -1
        )
        coords = torch.reshape(coords, [-1, 2])
        selected_indices = np.random.choice(coords.shape[0], size=[args.batch_size_random_rays], replace=False)
        selected_coords = coords[selected_indices]

        rays_o, rays_d = create_rays(h, w, intrinsic_camera_transformation, camera_to_world_transformation)
        rays_o = rays_o[selected_coords[:, 0], selected_coords[:, 1]]
        rays_d = rays_d[selected_coords[:, 0], selected_coords[:, 1]]

        rgb_map_coarse, rgb_map_fine = render(nerf, rays_o, rays_d, near=2, far=6, noise_std=0.)

        optimizer.zero_grad()

        image_rgb = image[selected_coords[:, 0], selected_coords[:, 1]]
        loss_coarse = mse(rgb_map_coarse, image_rgb)
        loss_fine = mse(rgb_map_fine, image_rgb)
        loss = loss_coarse + loss_fine

        psnr = mse2psnr(loss)

        loss.backward()
        optimizer.step()

        decay_steps = args.learning_rate_decay_steps * 1000
        decay_rate = 0.1
        new_l_rate = args.learning_rate_decay_steps * (decay_rate ** (step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_l_rate

        if step % args.step_ckpt == 0:
            path = os.path.join("./ckpts", args.name, '{:06d}.tar'.format(step))
            torch.save(
                {
                    'step': step,
                    'coarse_state': nerf['coarse_model'].state_dict(),
                    'fine_state': nerf['fine_model'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                },
                path
            )

        if step % args.step_video == 0:
            pass
            # # Turn on testing mode
            # with torch.no_grad():
            #     rgbs, disps = render_path(render_poses, hwf, K, args.chunk, render_kwargs_test)
            # print('Done, saving', rgbs.shape, disps.shape)
            # moviebase = os.path.join(basedir, expname, '{}_spiral_{:06d}_'.format(expname, i))
            # imageio.mimwrite(moviebase + 'rgb.mp4', to8b(rgbs), fps=30, quality=8)

        if step % args.step_print == 0:
            tqdm.write(f"[TRAIN] Step: {step}; Loss: {loss.item()}; PSNR: {psnr.item()}")


if __name__ == '__main__':
    args = parse_args()

    log_dir = f"logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    log_file = f"logs/{args.name}.txt"
    with open(log_file, 'w') as file:
        file.truncate()
    logging.basicConfig(filename=f"logs/{args.name}.txt", level=logging.INFO)

    torch.set_default_dtype(torch.float64)
    train()
