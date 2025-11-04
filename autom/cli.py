"""CLI for the autom command-executing AI bot."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from .bot import AutomBot

app = typer.Typer(help="Run the Autom command-executing AI bot.")


def _read_prompts(prompt: Optional[str], file: Optional[Path]) -> list[str]:
    prompts: list[str] = []
    if prompt:
        prompts.append(prompt)
    if file:
        prompts.extend(line.strip() for line in file.read_text().splitlines() if line.strip())
    if not prompts:
        raise typer.BadParameter("Provide either --prompt or --file with at least one message.")
    return prompts


@app.command()
def chat(
    prompt: Optional[str] = typer.Option(None, help="Initial prompt for the bot."),
    file: Optional[Path] = typer.Option(
        None, exists=True, file_okay=True, dir_okay=False, readable=True, help="File with prompts"
    ),
    model: str = typer.Option("gpt-4.1-mini", help="OpenAI model to use."),
    timeout: int = typer.Option(30, help="Shell command timeout in seconds."),
    max_turns: int = typer.Option(8, help="Maximum model-tool exchange loops per prompt."),
) -> None:
    """Run Autom against the provided prompts."""

    prompts = _read_prompts(prompt, file)
    bot = AutomBot(model=model, shell_timeout=timeout, max_turns=max_turns)
    bot.run_conversation(prompts)


@app.command()
def quote(*command: str) -> None:
    """Utility to preview how a command will be escaped."""

    from .bot import quote_command

    typer.echo(quote_command(command))


if __name__ == "__main__":
    app()
