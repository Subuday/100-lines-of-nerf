import json
import imageio.v2 as imageio
import configargparse
import os
import cv2
import numpy as np
import torch
from tqdm import trange


def create_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
    parser.add_argument("--data_dir", type=str, default='data/lego')
    parser.add_argument("--learning_rate_decay_steps", type=int, default=250,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--num_coarse_samples", type=int, default=64, help='number of coarse samples per ray')
    parser.add_argument("--num_fine_samples", type=int, default=0, help='number of fine samples per ray')
    parser.add_argument("--batch_size_random_rays", type=int, default=1024, help='number of random rays per batch')
    parser.add_argument("--use_reduced_resolution", action='store_true', help='use reduced resolution for training')
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
            fileName = os.path.join(args.data_dir, frame['file_path'] + '.png')
            image = imageio.imread(fileName)
            slit_images.append(image)

        split_transformations = []
        for frame in meta['frames']:
            split_transformations.append(np.array(frame['transform_matrix']))

        slit_images = (np.array(slit_images) / 255.).astype(np.float32)
        images.append(slit_images)

        split_transformations = np.array(split_transformations).astype(np.float32)
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
        for i, img in enumerate(reduced_images):
            reduced_images[i] = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        images = reduced_images

    return {
        'images': images,
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

    start = 1
    iters = 200_000 + 1
    for i in trange(start, iters):
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
        batch_rays = torch.stack([rays_o, rays_d], 0)


if __name__ == '__main__':
    args = parse_args()
    train()
