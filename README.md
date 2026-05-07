# Maintainability-Aware Code Generation Pipeline (MACGPi)

MACGPi is a maintainability-focused, LLM-powered code generation pipeline designed to produce production-quality, testable, and well-documented software artifacts. It is built to run with the `mini-swe-agent` orchestration and integrates prompt templates, phase schemas, and validation tooling to guide generation, evaluation, and iteration.

MACGPi requires a separate vLLM server running to communicate with. This is not orchestrated by the architecture.

## Key Features
- Maintainability-first prompts and schemas for reproducible artifact generation
- Phase-based pipeline (planning, implementation, evaluation) with JSON schemas
- Templates and evaluators to produce structured outputs and machine-validated results
- Lightweight integration with batch runners and local experimentation

## Repository Layout
  - `macgpi/` — core package and prompts used by the pipeline
  - `prompts/` — phase templates and JSON schemas.
  - `configs/` — configuration files.

See the prompts directory for examples and canonical templates:
- [prompts/01_plan/schema.json](prompts/01_plan/schema.json)
- [prompts/03_evaluate/template.md](prompts/03_evaluate/template.md)
- [prompts/03_evaluate/schema.json](prompts/03_evaluate/schema.json)

## Getting Started

Prerequisites
- Python 3.10+ recommended (or the project's specified environment)
- Conda/Anaconda is commonly used in this workspace for environment management

Quick local steps
1. Create or activate a Python environment (optional):

```bash
conda create -n macgpi python=3.10 -y
conda activate macgpi
```

2. Install project dependencies (if a `requirements.txt` or `pyproject.toml` exists):

```bash
pip install -r requirements.txt
```

3. Ensure a vLLM server is running with the required LLM.

3. Run MACGPi

```bash
python3 -m machpi <problem description> <model name> <output directory> [options] 
```

Adjust scripts and commands to your environment if you are not on Slurm.

## CLI parameters

### Required parameters

  - **input** (**str**): The input prompt/problem description which MACGPI should solve as a string.
  - **model_name** (**str**): The model name of an LLM that the vLLM server is hosting. A list of models for your vLLM server can be found by calling `GET <model_host>:<model_port>/v1/models`.
  - **output_dir** (**str**): The destination of the produces code and documentation artifacts.

### Optional parameters
 
  - **--model-host** (**str**): The hostname of the local vLLM server (default "localhost"). 
  - **--model-port** (**int**): The port of the local vLLM server (default 8000).
  - **--phases** (**list[str]**): The phases that should be run as a list of strings (default all phases).
  - **--prompt-dir** (**str**): The directory in which the phases and their corresponding prompts/schemas are located (default `macgpi/prompts`).
  - **--model-config** (**str**): The path to the config file for the LLM (default `macgpi/configs/model.config.yaml`, containing the default mini-swe-agent model config). See [this page](https://mini-swe-agent.com/latest/advanced/yaml_configuration/#model-configuration) for more information.
  - **--agent-config** (**str**): The path to the config file for the agent (default `macgpi/configs/agent.config.yaml`, containing the default mini-swe-agent agent config). See [this page](https://mini-swe-agent.com/latest/advanced/yaml_configuration/#agent-configuration) for more information.
  - **--log-level** (**str**): The verbosity of MACGPi and underlying models. Either `DEBUG`, `INFO`, `WARNING`, or `ERROR` (default `INFO`)
  
## Working with Prompts and Schemas
- Templates live in `prompts/` and are paired with machine-readable schemas to constrain output.
- When authoring a new phase, add both a `template.md` and a `schema.json` and reference them from `configs/macgpi_phases.json`.

