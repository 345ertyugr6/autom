# autom

Autom is a command-executing AI bot that uses the OpenAI Responses API alongside a shell
executor. It can receive user prompts, decide when to call the shell through a structured
function call, and return summaries of the executed commands.

## Features

- Built on the `openai` Python SDK with the Responses API
- Supports a `run_shell` function-tool for direct command execution
- CLI built with [Typer](https://typer.tiangolo.com/) offering chat and command quoting helpers
- Configurable model, shell timeout, and max tool-call loops per prompt

## Installation

```bash
pip install -e .
```

Set your OpenAI API key before running the bot:

```bash
export OPENAI_API_KEY="sk-..."
```

## Usage

### Single prompt

```bash
autom chat --prompt "List files in the repository"
```

### Multiple prompts

Create a text file (one prompt per line) and run:

```bash
autom chat --file prompts.txt
```

### Quote a command

```bash
autom quote python -m http.server --bind 0.0.0.0
```

## Extending

- Customize the system prompt by instantiating `AutomBot` directly and overriding
  `system_prompt`.
- Add more tools by extending `AutomBot.tool_specification()` and handling
  additional tool calls similar to `run_shell`.
