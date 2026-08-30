"""src/app/app.py - Gradio chat UI for JARVIS on Databricks App."""
import os
import gradio as gr
from databricks.sdk import WorkspaceClient
from databricks.sdk.core import Config
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

SERVING_ENDPOINT = os.environ.get("SERVING_ENDPOINT", "mini-jarvis-agent-endpoint")
_base_cfg = Config()


def _ws(user_token: str | None) -> WorkspaceClient:
    if user_token:
        return WorkspaceClient(host=_base_cfg.host, token=user_token, auth_type="pat")
    return WorkspaceClient(config=_base_cfg)


def respond(message, history, request: gr.Request):
    user_token = None
    if request:
        user_token = request.headers.get("x-forwarded-access-token")
    messages = []
    for turn in (history or []):
        if isinstance(turn, dict):
            if turn.get("content"):
                messages.append({"role": turn.get("role", "user"), "content": turn["content"]})
        else:
            u, a = turn
            if u: messages.append({"role": "user", "content": u})
            if a: messages.append({"role": "assistant", "content": a})
    messages.append({"role": "user", "content": message})
    try:
        ws = _ws(user_token)
        sdk_msgs = [ChatMessage(role=ChatMessageRole.USER if m["role"] == "user" else ChatMessageRole.ASSISTANT, content=m["content"]) for m in messages]
        resp = ws.serving_endpoints.query(name=SERVING_ENDPOINT, messages=sdk_msgs, max_tokens=800)
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error reaching JARVIS endpoint: {e}"


demo = gr.ChatInterface(
    fn=respond,
    title="JARVIS - AI Assistant",
    description="Powered by Gemini 3.6 Flash on Databricks. Ask about time, weather, math, or anything!",
    examples=["What time is it in Tokyo?", "What is 256 * 16?", "What is the weather in London?"],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    port = int(os.environ.get("DATABRICKS_APP_PORT", "8000"))
    demo.launch(server_name="0.0.0.0", server_port=port)