import matplotlib

from exper import parser
from exper.train import train

matplotlib.use("Agg")


def main():
    args = parser.get_parser().parse_args()
    args = vars(args)  # Converting the argparse Namespace to a dict

    if args["seed_range"] is not None:
        args["seed"] = list(range(args["seed_range"][0], args["seed_range"][1] + 1))
        print("Seed range:", args["seed"])

    for _ in train(args):
        # -> training according to the (<seed list> in) args
        pass


if __name__ == "__main__":
    main()
