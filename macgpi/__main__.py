import argparse
import logging

from macgpi.engine.main import MACGPi


def cli() -> int:
    parser = argparse.ArgumentParser(
        description="MACGPi: The Maintainability-Aware Code Generation Pipeline."
    )
    parser.add_argument("input_description", help="Project/issue description.")
    parser.add_argument(
        "model_name", help="Model name understood by mini-swe-agent.")
    parser.add_argument(
        "output_dir", help="Path to the target project directory.")
    parser.add_argument(
        "--model-host",
        default="localhost",
        help="Host for the model server.",
    )
    parser.add_argument(
        "--model-port",
        type=int,
        default=8000,
        help="Port for the model server.",
    )
    parser.add_argument(
        "--prompt-dir",
        default=None,
        help="The directory under which the valid phase prompts and schemas are located. If not provided, it defaults"
             + " to a 'prompts' directory located in the parent directory of this module.",
    )
    parser.add_argument(
        "--model-config-file",
        default=None,
        help="Configuration file for the model. If not provided, the default mini-swe-agent configuration will be "
             + "used.",
    )
    parser.add_argument(
        "--agent-config-file",
        default=None,
        help="Configuration file for the agent. If not provided, the default mini-swe-agent configuration will be "
        + "used.",
    )
    parser.add_argument(
        "--phases-config-file",
        default=None,
        help="Configuration file for the phases. If not provided, the default MACGPi configuration will be used.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="The verbosity of MACGPi, either DEBUG, INFO, WARNING, ERROR or CRITICAL."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Remove log_level from params since it's not used in macgpi function
    params = vars(args)
    params.pop("log_level")

    macgpi: MACGPi = MACGPi(**params)
    return 0 if macgpi.run() else 1


if __name__ == "__main__":
    exit(cli())
