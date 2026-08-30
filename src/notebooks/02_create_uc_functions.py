# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Create Unity Catalog Tool Functions
# MAGIC
# MAGIC Registers SQL functions in UC that the JARVIS agent can call as tools.

# COMMAND ----------
catalog = dbutils.widgets.get("catalog") if dbutils.widgets.getAll() else "agentic_catalog"
schema  = dbutils.widgets.get("schema")  if dbutils.widgets.getAll() else "mini_jarvis"

# COMMAND ----------
# MAGIC %md ## Tool: get_current_time
spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_current_time(timezone STRING DEFAULT 'UTC')
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Returns the current date and time for the given IANA timezone.'
AS $$
from datetime import datetime
from zoneinfo import ZoneInfo
try:
    tz = ZoneInfo(timezone)
except Exception:
    tz = ZoneInfo('UTC')
    timezone = 'UTC'
now = datetime.now(tz)
return f"Current time in {{timezone}}: {{now.strftime('%A, %B %d %Y %I:%M %p %Z')}}"
$$
""")
print("get_current_time registered")

# COMMAND ----------
# MAGIC %md ## Tool: get_weather
spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.get_weather(city STRING)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Returns a weather report for a city (stub - replace with real API).'
AS $$
mock = {{'New York': '72F, partly cloudy', 'London': '58F, overcast', 'Tokyo': '81F, sunny'}}
report = mock.get(city, '22C, clear skies')
return f"Weather in {{city}}: {{report}}"
$$
""")
print("get_weather registered")

# COMMAND ----------
# MAGIC %md ## Tool: calculate
spark.sql(f"""
CREATE OR REPLACE FUNCTION {catalog}.{schema}.calculate(expression STRING)
RETURNS STRING
LANGUAGE PYTHON
COMMENT 'Evaluates a simple arithmetic expression.'
AS $$
allowed = set('0123456789+-*/()., ')
if not all(c in allowed for c in expression):
    return 'Error: unsafe expression.'
try:
    result = eval(expression, {{'__builtins__': {{}}}})
    return f"{{expression}} = {{result}}"
except Exception as e:
    return f"Calculation error: {{e}}"
$$
""")
print("calculate registered")

# COMMAND ----------
print(f"All UC tools registered in {catalog}.{schema}")