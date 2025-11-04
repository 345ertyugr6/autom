"""Core logic for the autom command-executing AI bot."""

from __future__ import annotations

import json
import shlex
import subprocess
import textwrap
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from openai import OpenAI


@dataclass
class ShellResult:
    """Represents the outcome of a shell command execution."""

    command: str
    returncode: int
    stdout: str
    stderr: str

    def to_tool_message(self) -> str:
        """Format the shell result for the model to consume."""

        body = textwrap.dedent(
            f"""
            Command: {self.command}
            Exit code: {self.returncode}
            Stdout:\n{self.stdout or '<empty>'}
            Stderr:\n{self.stderr or '<empty>'}
            """
        ).strip()
        return body


class ShellExecutor:
    """Run shell commands with a configurable timeout."""

    def __init__(self, timeout: int = 30) -> None:
        self.timeout = timeout

    def run(self, command: str) -> ShellResult:
        """Execute *command* in a shell and return its results."""

        completed = subprocess.run(
            command,
            shell=True,
            check=False,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return ShellResult(
            command=command,
            returncode=completed.returncode,
            stdout=completed.stdout.strip(),
            stderr=completed.stderr.strip(),
        )


class AutomBot:
    """A lightweight orchestrator around the OpenAI Responses API with shell access."""

    def __init__(
        self,
        *,
        model: str = "gpt-4.1-mini",
        system_prompt: str | None = None,
        shell_timeout: int = 30,
        max_turns: int = 8,
    ) -> None:
        self.client = OpenAI()
        self.model = model
        self.shell = ShellExecutor(timeout=shell_timeout)
        self.max_turns = max_turns
        self._system_prompt = system_prompt or (
            "You are Autom, an assistant that can solve tasks by running shell commands."
            " Use shell responsibly and summarize the results for the user."
        )

    def tool_specification(self) -> List[dict]:
        """Describe the available tools for the Responses API."""

        return [
            {
                "type": "function",
                "function": {
                    "name": "run_shell",
                    "description": "Execute a shell command on the local machine.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Command to execute, passed directly to the shell.",
                            }
                        },
                        "required": ["command"],
                    },
                },
            }
        ]

    def _build_messages(self, conversation: Sequence[dict]) -> List[dict]:
        messages: List[dict] = [
            {"role": "system", "content": self._system_prompt}
        ]
        messages.extend(conversation)
        return messages

    def _handle_tool_call(self, arguments: str) -> dict:
        payload = json.loads(arguments)
        command = payload["command"]
        shell_result = self.shell.run(command)
        return {
            "role": "tool",
            "content": shell_result.to_tool_message(),
            "name": "run_shell",
        }

    def run_conversation(self, prompts: Iterable[str]) -> None:
        """Run an interactive conversation for each user prompt provided."""

        conversation: List[dict] = []
        for user_input in prompts:
            conversation.append({"role": "user", "content": user_input})
            for _ in range(self.max_turns):
                response = self.client.responses.create(
                    model=self.model,
                    messages=self._build_messages(conversation),
                    tools=self.tool_specification(),
                )
                tool_messages: List[dict] = []
                printed_output = False
                for item in response.output:
                    if item.type == "message":
                        printed_output = True
                        content_parts = item.content
                        message_text = "\n".join(
                            part.text for part in content_parts if part.type == "output_text"
                        )
                        if message_text:
                            print(message_text)
                            conversation.append({"role": item.role, "content": message_text})
                    elif item.type == "tool_call":
                        function_call = item.function
                        if function_call and function_call.name == "run_shell":
                            tool_message = self._handle_tool_call(function_call.arguments or "{}")
                            tool_messages.append(tool_message)
                            conversation.append(tool_message)
                if tool_messages:
                    continue
                if not printed_output:
                    print("(No response from model)")
                break

    def run_prompt(self, prompt: str) -> None:
        """Convenience wrapper to run a single prompt."""

        self.run_conversation([prompt])


def quote_command(command: Sequence[str]) -> str:
    """Join a sequence of command parts into a shell-escaped string."""

    return " ".join(shlex.quote(part) for part in command)
