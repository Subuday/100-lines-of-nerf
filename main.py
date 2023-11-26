import configargparse


def create_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, help='config file path')
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


if __name__ == '__main__':
    args = parse_args()
    pass
