import yaml
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    with open(args.config, 'r') as stream:
        try:
            param = yaml.safe_load(stream)
        except yaml.YAMLError:
            exit(0)

    args = vars(args)  # Converting argparse Namespace to a dict.
    args.update(param)  # Add parameters from config file
    
    train(args)


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    parser.add_argument('--config', type=str, default='',
                        help='Yaml file of settings.')
    parser.add_argument('--seed', type=int, default=1993)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--init_lr', type=float, default=0.001)
    parser.add_argument('--tuned_epoch', type=int, default=5)
    parser.add_argument('--memory_per_class', type=int, default=20)
    parser.add_argument('--init_cls', type=int, default=10)
    parser.add_argument('--increment', type=int, default=10)
    parser.add_argument('--pretrained_weight', type=str, default='laion400m_e32')
    parser.add_argument('--pool_size', type=int, default=100)
    parser.add_argument('--prompt_depth_text', type=int, default=12)
    parser.add_argument('--prompt_depth_vision', type=int, default=12)
    parser.add_argument('--n_ctx_vision', type=int, default=4)
    parser.add_argument('--n_ctx_text', type=int, default=4)
    parser.add_argument('--alpha_div', type=float, default=1)
    parser.add_argument('--gadi', action='store_true', default=False,
                        help='config on gadi machine')
    parser.add_argument('--local_path', type=str)
    
    return parser


if __name__ == '__main__':
    main()
