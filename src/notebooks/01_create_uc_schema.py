# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Create UC Schema + Gemini External Model Endpoint
# MAGIC
# MAGIC Creates:
# MAGIC - Unity Catalog schema: agentic_catalog.mini_jarvis
# MAGIC - Gemini 3.6 Flash external model endpoint (OpenAI-compat)

# COMMAND ----------
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    ServedEntityInput,
    EndpointCoreConfigInput,
    ExternalModel,
    ExternalModelProvider,
    OpenAiConfig,
)

ws = WorkspaceClient()
catalog = dbutils.widgets.get("catalog") if dbutils.widgets.getAll() else "agentic_catalog"
schema  = dbutils.widgets.get("schema")  if dbutils.widgets.getAll() else "mini_jarvis"
llm_ep  = dbutils.widgets.get("llm_endpoint") if dbutils.widgets.getAll() else "gemini-3-6-flash-endpoint"

# COMMAND ----------
# MAGIC %md ## 1. Create schema
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
print(f"Schema ready: {catalog}.{schema}")

# COMMAND ----------
# MAGIC %md ## 2. Create Gemini External Model Endpoint
gemini_api_key = dbutils.secrets.get("mini-jarvis", "GEMINI_API_KEY")

try:
    existing = ws.serving_endpoints.get(llm_ep)
    print(f"Endpoint {llm_ep!r} already exists: {existing.state}")
except Exception:
    print(f"Creating endpoint {llm_ep!r}...")
    ws.serving_endpoints.create_and_wait(
        name=llm_ep,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    name="gemini-3-6-flash",
                    external_model=ExternalModel(
                        name="gemini-2.0-flash",
                        provider=ExternalModelProvider.OPENAI,
                        task="llm/v1/chat",
                        openai_config=OpenAiConfig(
                            openai_api_key_plaintext=gemini_api_key,
                            openai_api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
                        ),
                    ),
                )
            ]
        ),
    )
    print(f"Endpoint {llm_ep!r} created.")