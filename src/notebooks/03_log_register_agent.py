# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Build, Log & Register the JARVIS Agent
# MAGIC
# MAGIC Mirrors the pattern from databricks-agentic-ai/06_log_register_agent.py
# MAGIC - Builds a tool-calling agent using the Gemini external model endpoint
# MAGIC - Logs with MLflow, registers in Unity Catalog model registry

# COMMAND ----------
catalog      = dbutils.widgets.get("catalog")      if dbutils.widgets.getAll() else "agentic_catalog"
schema       = dbutils.widgets.get("schema")       if dbutils.widgets.getAll() else "mini_jarvis"
llm_endpoint = dbutils.widgets.get("llm_endpoint") if dbutils.widgets.getAll() else "gemini-3-6-flash-endpoint"
model_name   = dbutils.widgets.get("model_name")   if dbutils.widgets.getAll() else f"{catalog}.{schema}.jarvis_agent"

# COMMAND ----------
# Build agent source — same ToolCallingAgent pattern as existing repo
AGENT_TEMPLATE = '''
import json
import warnings
from uuid import uuid4
from typing import Any, Generator

import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import FunctionInfo
from mlflow.types.agent import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.models.rag_signatures import StringResponse
from openai import OpenAI

mlflow.openai.autolog()

DATABRICKS_HOST = WorkspaceClient().config.host
LLM_ENDPOINT = "__LLM_ENDPOINT__"
CATALOG = "__CATALOG__"
SCHEMA  = "__SCHEMA__"

SYSTEM_PROMPT = """You are JARVIS, a helpful AI assistant deployed on Databricks.
You have tools to get the current time, check weather, and do calculations.
Be concise and friendly."""

client = OpenAI(
    base_url=f"{DATABRICKS_HOST}/serving-endpoints/{LLM_ENDPOINT}/invocations",
    api_key="token",
    default_headers={"Authorization": f"Bearer {WorkspaceClient().config.token}"},
)

UC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Returns current date/time for a timezone.",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "IANA timezone, e.g. America/New_York"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Returns weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a simple arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "Arithmetic expression"}},
                "required": ["expression"],
            },
        },
    },
]


def execute_uc_function(name: str, args: dict) -> str:
    ws = WorkspaceClient()
    result = ws.statement_execution.execute_statement(
        warehouse_id=ws.config.cluster_id or "auto",
        statement=f"SELECT {CATALOG}.{SCHEMA}.{name}({', '.join(repr(v) for v in args.values())})",
        wait_timeout="30s",
    )
    rows = result.result.data_array if result.result else []
    return rows[0][0] if rows else "No result"


class JarvisAgent(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params=None):
        if isinstance(model_input, dict):
            messages = model_input.get("messages", [])
        else:
            messages = [{"role": "user", "content": str(model_input)}]

        history = [{"role": "system", "content": SYSTEM_PROMPT}] + messages
        for _ in range(8):
            resp = client.chat.completions.create(
                model=LLM_ENDPOINT,
                messages=history,
                tools=UC_TOOLS,
            )
            msg = resp.choices[0].message
            history.append(msg.model_dump())

            if not msg.tool_calls:
                return {"response": msg.content}

            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                result = execute_uc_function(tc.function.name, args)
                history.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

        return {"response": "Max reasoning steps reached."}


mlflow.pyfunc.set_model(JarvisAgent())
'''

agent_src = (
    AGENT_TEMPLATE
    .replace("__LLM_ENDPOINT__", llm_endpoint)
    .replace("__CATALOG__", catalog)
    .replace("__SCHEMA__", schema)
)
with open("jarvis_agent.py", "w") as f:
    f.write(agent_src)
print("Wrote jarvis_agent.py")

# COMMAND ----------
import mlflow
from databricks.sdk import WorkspaceClient
from mlflow.models.resources import DatabricksServingEndpoint, DatabricksFunction
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
username = WorkspaceClient().current_user.me().user_name
mlflow.set_experiment(f"/Users/{username}/mini_jarvis")

resources = [
    DatabricksServingEndpoint(endpoint_name=llm_endpoint),
    DatabricksFunction(function_name=f"{catalog}.{schema}.get_current_time"),
    DatabricksFunction(function_name=f"{catalog}.{schema}.get_weather"),
    DatabricksFunction(function_name=f"{catalog}.{schema}.calculate"),
]

input_example = {"messages": [{"role": "user", "content": "What time is it in Tokyo?"}]}

with mlflow.start_run():
    info = mlflow.pyfunc.log_model(
        name="jarvis_agent",
        python_model="jarvis_agent.py",
        resources=resources,
        input_example=input_example,
        pip_requirements=["databricks-sdk", "openai"],
        registered_model_name=model_name,
    )
print(f"Logged & registered: {info.model_uri}")

# COMMAND ----------
client = MlflowClient(registry_uri="databricks-uc")
versions = client.search_model_versions(f"name='{model_name}'")
latest = max(versions, key=lambda v: int(v.version)).version
print(f"Registered version: {latest}")
dbutils.jobs.taskValues.set(key="model_version", value=str(latest))