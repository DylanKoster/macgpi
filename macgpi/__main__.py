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

    parser.add_argument(
        "--model-toolset",
        default=None,
        help="The tool call parser for the model. If None, the tool call parser will tried to be implied, if unsuccesfull, an error will occur. See https://docs.vllm.ai/en/latest/features/tool_calling/#automatic-function-calling",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="The maximum context length for the model. If None, the default context length of the model will be used.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="The amount of multithreading to use.",
    )
    parser.add_argument(
        "--phases",
        nargs="+",
        default=None,
        help="The phases of the pipeline to execute, in order. If not provided, all phases will be executed in the order they are found in the prompts directory.",
    )
    parser.add_argument(
        "--prompt-dir",
        default=None,
        help="The directory under which the valid phase prompts and schemas are located. If not provided, it defaults to a 'prompts' directory located in the parent directory of this module.",
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Configuration file for the model. If not provided, the default mini-swe-agent configuration will be used.",
    )
    parser.add_argument(
        "--agent-config",
        default=None,
        help="Configuration file for the agent. If not provided, the default mini-swe-agent configuration will be used.",
    )

    args = parser.parse_args()

    macgpi(**vars(args))

if __name__ == "__main__":
    cli()