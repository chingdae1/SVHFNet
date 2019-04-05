import argparse
import yaml
from solver import Solver


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str, default='./option/baseline.yaml')

    args = parser.parse_args()

    with open(args.config_path, 'r') as config:
        config = yaml.load(config.read())
    solver = Solver(config)
    solver.fit()

if __name__ == '__main__':
    main()
