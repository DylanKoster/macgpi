import argparse

from macgpi.engine.main import macgpi

def cli():
    parser = argparse.ArgumentParser(
        description="MACGPi: The Maintainability-Aware Code Generation Pipeline."
    )
    parser.add_argument("input", help="Project/issue description.")
    parser.add_argument("model_name", help="Model name understood by mini-swe-agent.")
    parser.add_argument("output_dir", help="Path to the target project directory.")
    parser.add_argument(
        "--host-vllm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True, MACGPi hosts the vLLM itself; if False, provide --model-host and --model-port.",
    )
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

    args = parser.parse_args()

    macgpi(**vars(args))

if __name__ == "__main__":
    cli()