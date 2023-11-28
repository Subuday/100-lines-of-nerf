import json
import imageio.v2 as imageio
import configargparse
import os
import cv2
import numpy as np
import torch


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


if __name__ == '__main__':
    args = parse_args()
    images, transformations, camera_poses, hwf, splits = load_data()

    pass
