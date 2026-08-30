# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Deploy JARVIS Agent to Model Serving Endpoint

# COMMAND ----------
catalog        = dbutils.widgets.get("catalog")        if dbutils.widgets.getAll() else "agentic_catalog"
schema         = dbutils.widgets.get("schema")         if dbutils.widgets.getAll() else "mini_jarvis"
model_name     = dbutils.widgets.get("model_name")     if dbutils.widgets.getAll() else f"{catalog}.{schema}.jarvis_agent"
agent_endpoint = dbutils.widgets.get("agent_endpoint") if dbutils.widgets.getAll() else "mini-jarvis-agent-endpoint"

try:
    model_version = dbutils.jobs.taskValues.get(taskKey="log_register_agent", key="model_version")
except Exception:
    from mlflow.tracking import MlflowClient
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()
    versions = client.search_model_versions(f"name='{model_name}'")
    model_version = max(versions, key=lambda v: int(v.version)).version

print(f"Deploying {model_name} v{model_version} to endpoint {agent_endpoint!r}")

# COMMAND ----------
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)

ws = WorkspaceClient()

config = EndpointCoreConfigInput(
    served_entities=[
        ServedEntityInput(
            entity_name=model_name,
            entity_version=str(model_version),
            name=f"{model_name.replace('.', '-')}-{model_version}",
            scale_to_zero_enabled=True,
        )
    ],
    traffic_config=TrafficConfig(
        routes=[Route(served_model_name=f"{model_name.replace('.', '-')}-{model_version}", traffic_percentage=100)]
    ),
)

try:
    ws.serving_endpoints.get(agent_endpoint)
    ws.serving_endpoints.update_config_and_wait(name=agent_endpoint, served_entities=config.served_entities)
    print(f"Endpoint {agent_endpoint!r} updated.")
except Exception:
    ws.serving_endpoints.create_and_wait(name=agent_endpoint, config=config)
    print(f"Endpoint {agent_endpoint!r} created.")

# COMMAND ----------
# Smoke test
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
resp = ws.serving_endpoints.query(
    name=agent_endpoint,
    messages=[ChatMessage(role=ChatMessageRole.USER, content="What time is it in UTC?")],
)
print("Smoke test response:", resp.choices[0].message.content)