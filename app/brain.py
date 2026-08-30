"""app/brain.py — Gemini-powered tool-calling agent using google-genai SDK."""
from __future__ import annotations

import os
from google import genai
from google.genai import types

from app.config import settings
from app.tools import TOOL_SPECS, dispatch

SYSTEM_PROMPT = """You are JARVIS, a helpful and witty AI assistant.
You have access to tools for checking the time, weather, setting reminders, and calculations.
Always be concise. When using a tool, briefly explain what you are doing."""

_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set. Add it to .env")
        _client = genai.Client(api_key=api_key)
    return _client


def _build_tool_config() -> list[types.Tool]:
    """Convert our TOOL_SPECS list to google-genai Tool format."""
    declarations = []
    for spec in TOOL_SPECS:
        props = {}
        for name, info in spec["parameters"]["properties"].items():
            ptype = types.Type.STRING if info["type"] == "string" else types.Type.INTEGER
            props[name] = types.Schema(type=ptype, description=info.get("description", ""))

        declarations.append(
            types.FunctionDeclaration(
                name=spec["name"],
                description=spec["description"],
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties=props,
                    required=spec["parameters"].get("required", []),
                ),
            )
        )
    return [types.Tool(function_declarations=declarations)]


class JarvisBrain:
    def __init__(self):
        self._history: list[types.Content] = []
        self._tools = _build_tool_config()
        self.tool_log: list[dict] = []  # [{timestamp, tool, args, result}]

    def chat(self, user_message: str) -> str:
        """Send a message and return JARVIS response (handles tool calls internally)."""
        client = _get_client()
        model = settings.gemini_model

        self._history.append(
            types.Content(role="user", parts=[types.Part(text=user_message)])
        )

        for _ in range(8):  # max tool-call iterations
            response = client.models.generate_content(
                model=model,
                contents=self._history,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT,
                    tools=self._tools,
                ),
            )
            candidate = response.candidates[0]
            self._history.append(candidate.content)

            # Collect function calls from all parts
            fn_calls = [
                p.function_call
                for p in candidate.content.parts
                if p.function_call is not None
            ]

            if not fn_calls:
                # Final text response
                text = "".join(
                    p.text for p in candidate.content.parts
                    if hasattr(p, "text") and p.text
                )
                return text.strip()

            # Execute tools and push results back as a user turn
            tool_parts = []
            import datetime
            for fc in fn_calls:
                args = dict(fc.args) if fc.args else {}
                result = dispatch(fc.name, args)
                timestamp = datetime.datetime.now().strftime("%H:%M:%S")
                self.tool_log.append({
                    "timestamp": timestamp,
                    "tool": fc.name,
                    "args": args,
                    "result": result,
                })
                print(f"  [tool] {fc.name}({args}) -> {result}")
                tool_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": result},
                        )
                    )
                )

            self._history.append(
                types.Content(role="user", parts=tool_parts)
            )

        return "I reached the maximum reasoning steps. Please try again."

    def reset(self):
        self._history = []
        self.tool_log = []


def run_cli():
    print("=" * 55)
    print("  JARVIS — Local Text Mode  (type 'quit' to exit)")
    print("=" * 55)
    brain = JarvisBrain()
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down. Goodbye!")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "bye"):
            print("JARVIS: Goodbye!")
            break
        response = brain.chat(user_input)
        print(f"JARVIS: {response}")


if __name__ == "__main__":
    run_cli()
