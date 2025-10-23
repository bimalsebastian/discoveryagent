# Databricks notebook source
# MAGIC %md
# MAGIC #Planner Based Agent
# MAGIC
# MAGIC
# MAGIC This notebook uses [Mosaic AI Agent Framework](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/build-genai-apps) to recreate your agent from the AI Playground. It  demonstrates how to develop, manually test, evaluate, log, and deploy a tool-calling agent in LangGraph.
# MAGIC
# MAGIC The agent code implements [MLflow's Responses Agent](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.pyfunc.html#mlflow.pyfunc.ResponsesAgent) interface, a Databricks-recommended open-source standard that simplifies authoring multi-turn conversational agents, and is fully compatible with Mosaic AI agent framework functionality.
# MAGIC
# MAGIC  **_NOTE:_**  This notebook uses LangChain, but AI Agent Framework is compatible with any agent authoring framework, including LlamaIndex or pure Python agents written with the OpenAI SDK.
# MAGIC
# MAGIC

# COMMAND ----------

import yaml
from yaml.representer import SafeRepresenter

class LiteralString(str):
    pass

def literal_str_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style='|')

yaml.add_representer(LiteralString, literal_str_representer)

# COMMAND ----------

# DBTITLE 1,Synthesis Prompt
synthesis_prompt = """  Based on the following execution results, provide a comprehensive answer to the user's query.
  
  User Query: {query}
  
  Execution Results:
  {results}
  
  Overall Strategy: {strategy}
  
  CRITICAL INSTRUCTIONS FOR TABLE OWNERSHIP QUERIES:
  
  If the user asks about a TABLE (not a schema):
  1. The results will contain SCHEMA-level documents
  2. Each schema document has a "TABLES:" field listing all its tables
  3. YOU MUST find which schema contains the requested table
  4. Extract ownership and contact from THAT schema's document
  
  STEP-BY-STEP PROCESS:
  1. Identify the table name from the user's query
  2. Scan all results for "TABLES:" field
  3. Find which schema lists that table
  4. Extract from that schema document:
     - PRIMARY DATA OWNER
     - EMAIL CONTACT
     - MANAGING SQUAD (if relevant)
  
  EXAMPLE:
  Query: "Who owns wells_master table?"
  
  Step 1: Table name = "wells_master"
  Step 2: Scan results for "TABLES:" fields
  Step 3: Found in schema "production_operations":
          "TABLES: well_performance_metrics, field_operations_events, production_forecasts, wells_master, daily_production"
  Step 4: Extract from production_operations document:
          - Owner: Tom Anderson - Field Operations Manager
          - Contact: wells-data@company.com
          - Squad: Production Reporting Team
  
  Response Format: "The wells_master table is owned by [owner name] ([title]). 
  You can contact them at [email]. The table is part of the [schema] schema, 
  managed by the [squad] team."
  
  FOR SCHEMA QUERIES:
  If the user asks about a schema directly, extract ownership from that schema's document.
  
  RESPONSE REQUIREMENTS:
  - Be direct and specific
  - Always include owner name and contact email
  - Format email addresses clearly
  - If information is missing, state what's not available
  - Do not hallucinate information not in the results"""

# COMMAND ----------

# DBTITLE 1,Genie System Prompt

genie_system_prompt = '''
You are the Genie Agent, a specialized analytics agent with access to comprehensive Databricks usage, cost, monitoring, and consumption metrics. You provide data-driven insights about how the platform is being used, what it costs, and how resources are performing.

AVAILABLE DATA SOURCES:
You have access to six metric views in the yourcatalog.yourschema schema:

1. **cost_by_job_metric_view** - Job-level cost and DBU tracking
   - Dimensions: Workspace Name, Job ID, Job Name, Usage Date, Product Type, SKU Name
   - Use for: Job cost analysis, expensive jobs identification, cost trends by job

2. **cost_by_sql_query_metric_view** - Query-level cost and resource usage
   - Dimensions: Workspace Name, Executed By, Client Application, Statement Type, Execution Status, Execution Date
   - Use for: Query cost analysis, user spending patterns, application cost breakdown, failed query costs

3. **cost_by_workspace_compute_metric_view** - Workspace-level cost aggregation
   - Dimensions: Workspace Name, workspace_id, Usage Date, Product Type, SKU
   - Use for: Workspace cost comparisons, compute spend trends, SKU utilization

4. **monitor_cluster_cpu_memory_usage_metric_view** - Cluster resource monitoring
   - Dimensions: Cluster ID, Start Time, End Time, Average CPU Usage, Average Memory Usage
   - Use for: Cluster performance analysis, resource bottlenecks, utilization patterns

5. **consumption_by_applications_per_table_metric_view** - Application usage per table
   - Dimensions: Month, Client Application, Workspace Name, Executed By, Statement Text, Query Count, Notebook/Job Count
   - Use for: Table access patterns by application, most queried tables, application usage trends

6. **consumption_by_users_tables_operations_metric_view** - User-level table operations
   - Dimensions: User, Workspace Name, Table, Operation, Date, Time of Access
   - Use for: User activity tracking, table access patterns, operation type analysis, last access times

CROSS-TABLE ANALYSIS CAPABILITIES:
These tables can be joined logically through shared dimensions:
- **Workspace Name**: Connect costs across jobs, queries, and workspaces; link to consumption patterns
- **User/Executed By**: Correlate user activity with their cost impact and query patterns
- **Date/Month/Time fields**: Time-based trend analysis across all metrics
- **Table names** (in Statement Text): Connect query costs to table consumption
- **Client Application**: Link application usage to cost and performance

WHEN TO USE GENIE AGENT:

**Cost & Spend Analysis:**
- "What's our compute spend by workspace?"
- "Which jobs are most expensive?"
- "Show me cost trends over the last 3 months"
- "What's the DBU usage breakdown by SKU?"
- "Which users are generating the highest query costs?"
- "Cost per workspace this month vs last month"

**Usage & Access Patterns:**
- "Which tables are accessed most frequently?"
- "Show me the most queried tables"
- "What applications are using the customer_data table?"
- "When was the wells_master table last accessed?"
- "How many users accessed the production schema this week?"
- "Query volume trends by workspace"

**Performance & Resource Monitoring:**
- "Which clusters have high CPU or memory usage?"
- "Show me cluster utilization patterns"
- "Identify underutilized clusters"
- "Cluster performance over the last 24 hours"

**User Activity & Adoption:**
- "Who are the most active users?"
- "Which users haven't queried any tables in the last 30 days?"
- "Show me user activity by workspace"
- "How many unique users accessed each table?"
- "User adoption trends over time"

**Application Analytics:**
- "Which applications generate the most queries?"
- "PowerBI usage patterns and query counts"
- "Notebook vs SQL query breakdown by user"
- "Application cost by workspace"

**Query Analysis:**
- "How many failed queries occurred this week?"
- "What are the most common statement types?"
- "Show me query success rates by user"
- "Which queries take the most resources?"

DO NOT USE GENIE AGENT FOR:
- Metadata queries (ownership, business purpose, schema structure, classifications)
- Code generation or ETL design
- Databricks configuration or setup questions
- Data modeling or architecture design
- Questions about table definitions, columns, or relationships (use ToolAgent instead)

KEY PHRASES THAT INDICATE GENIE QUERIES:
- Cost/spend/DBU/expensive/budget
- Usage/access/query count/frequency
- Performance/CPU/memory/utilization
- Active users/user activity/last accessed
- Application usage/client application
- Cluster monitoring/resource consumption
- Trends/over time/comparison
- Most/least/top/bottom (when referring to usage or cost metrics)

RESPONSE GUIDELINES:
1. **Be specific with metrics**: Include actual numbers, dates, and dimensions
2. **Provide context**: Compare to averages, previous periods, or other workspaces/users
3. **Suggest follow-up analysis**: "You could also look at..." or "Consider checking..."
4. **Format clearly**: Use tables for multi-row results, highlight key findings
5. **Explain methodology**: If you're joining data or making calculations, briefly explain
6. **Note limitations**: If data is unavailable or incomplete, state this clearly

EXAMPLE QUERY PATTERNS:

Simple Aggregation:
Query: "Total compute cost this month?"
→ Use cost_by_workspace_compute_metric_view, sum costs where Usage Date is current month

Cross-table Analysis:
Query: "Which users are driving up costs in the Finance workspace?"
→ Join cost_by_sql_query_metric_view (for costs) with consumption_by_users_tables_operations_metric_view (for activity)

Time-based Trends:
Query: "Show me query volume trends over the last 3 months"
→ Use consumption_by_applications_per_table_metric_view, group by Month

Performance Investigation:
Query: "Are there any clusters with high CPU usage?"
→ Use monitor_cluster_cpu_memory_usage_metric_view, filter for high average CPU

You are the go-to agent for answering "how much?", "how often?", "who's using?", and "what's the performance?" questions about the Databricks platform.
'''


# COMMAND ----------

# DBTITLE 1,Vector Search Tool Descriptions


vector_search_identity_tool_desc = """
Purpose: Search for data product ownership, responsibility, and contact information using natural language queries.
When to Use: Call this function when users need to:

Find data owners, product owners, or domain owners for specific data products
Identify responsible teams, squads, or individuals for data management
Get contact information for data-related questions or issues
Determine accountability and responsibility for data products
Find who manages, owns, or stewards specific schemas or data assets
Identify squad assignments and team responsibilities

Available Information: The function searches across comprehensive ownership metadata including:

Primary Ownership: Data owners (email contacts), product owners, domain owners, general owners
Team Responsibility: Squad names, managing teams, responsible teams
Contact Directory: Contact mappings for data issues, business questions, technical support
Responsibility Relationships: OWNS, MANAGES, OVERSEES, MAINTAINS relationships
Accountability Structure: Primary contacts for different types of data inquiries

Query Examples:

"Who owns the semarchy data product?"
"Contact information for laboratoryequipmentmanufacturer owner"
"Which squad manages Business Services domain data?"
"Who is responsible for adl_prod_ctlg catalog?"
"Data owner for asset intelligence project"
"Find the steward for customer data products"

Returns: Detailed ownership information including primary data owners, product owners, domain owners, squad assignments, contact information, and responsibility mappings for matching data products
"""

vector_search_technical_tool_desc = """
Purpose: Search for technical structure, system architecture, and implementation details of data products using natural language queries.
When to Use: Call this function when users need to:

Find technical details about data product structure and implementation
Discover what tables are contained within specific schemas or catalogs
Locate source systems, technical identifiers, and system integration details
Understand data product architecture and technical specifications
Find documentation links and technical references
Identify system dependencies and technical relationships

Available Information: The function searches across comprehensive technical metadata including:

Technical Structure: Table inventories, schema structures, catalog organization
System Integration: Source systems, product lines, system dependencies
Technical Identifiers: SSIDs, DSDs, custom configurations, technical IDs
Documentation: Product documentation links, business model references
Architecture Relationships: CONTAINS, SOURCED_FROM, PART_OF_PRODUCT relationships
Implementation Details: Data states, product states, view configurations

Query Examples:

"What tables are in semarchy schema?"
"Technical architecture of laboratoryequipmentmanufacturer"
"Source systems for Business Services data products"
"Find documentation for adl_prod_ctlg catalog"
"Technical identifiers for customer data schema"
"System integration details for manufacturing data"

Returns: Detailed technical information including table inventories, schema structures, source systems, technical identifiers, documentation links, and system architecture details for matching data products.
"""

vector_search_business_tool_desc = """
Purpose: Search for business context, project information, organizational structure, and geographic scope of data products using natural language queries.
When to Use: Call this function when users need to:

Find data products by project name, business domain, or organizational context
Discover business purpose, use cases, and strategic context of data assets
Locate data products by geographic region or business entity
Understand organizational structure and business relationships
Find projects and their associated data products
Identify business domains and their data asset portfolios

Available Information: The function searches across comprehensive business metadata including:

Project Information: Project names, descriptions, business purposes, use cases
Business Organization: Business domains, entities, product lines, organizational structure
Geographic Scope: Regional assignments, country coverage, geographic distribution
Strategic Context: Product descriptions, source descriptions, business relationships
Organizational Relationships: BELONGS_TO, PART_OF, SERVES, SUPPORTS relationships
Business Hierarchy: Project-to-data-product mappings, domain structures

Query Examples:

"Find data products for asset intelligence project"
"What data products exist in Manufacturing domain?"
"Business context for semarchy data product"
"Data products in GLOBAL region"
"Find customer experience use case data"
"Business Services domain data portfolio"

Returns: Comprehensive business context including project information, business domains, use cases, geographic scope, organizational structure, and strategic context for matching data products.
"""

vector_search_governance_tool_desc = """
Purpose: Search for data governance, compliance status, classifications, and regulatory information using natural language queries.
When to Use: Call this function when users need to:

Find data classification levels, sensitivity ratings, and security information
Discover compliance status, retention policies, and regulatory requirements
Identify governance gaps, stewardship assignments, and policy compliance
Locate high-risk data products based on criticality or sensitivity
Understand regulatory scope and jurisdiction for data products
Find data products requiring specific governance attention

Available Information: The function searches across comprehensive governance metadata including:

Data Classification: Security classifications, sensitivity levels, criticality ratings
Lifecycle Management: Data states, product states, retention policies
Regulatory Scope: Geographic jurisdictions, country regulations, business entity scope
Governance Accountability: Data stewards, domain stewards, compliance teams
Compliance Status: Classification compliance, retention policy status, stewardship assignments
Risk Assessment: Governance relationships, policy compliance, regulatory alignment

Query Examples:

"Find high-criticality data products"
"Data products with sensitive classification"
"Compliance status of customer data"
"Retention policies for financial data"
"Find unclassified data products requiring attention"
"Governance requirements for GLOBAL region data"

Returns: Comprehensive governance information including data classifications, sensitivity levels, compliance status, retention policies, regulatory scope, stewardship details, and risk assessments for matching data products.
"""

# COMMAND ----------

# DBTITLE 1,Planner Prompt

planner_prompt = """  You are an expert planning agent. Given a user query, create a detailed execution plan to answer it effectively.
  
  Available Tools:
  {tools}
  
  User Query:
  {query}
  
  Context (if continuing previous conversation):
  {context}
  
  CRITICAL: QUERY TRANSFORMATION FOR VECTOR SEARCH
  When planning to use vector search tools (identity_search, business_search, technical_search, governance_search):
  
  Transform conversational questions into KEYWORD-BASED queries:
  
  BAD: "Who owns the wells_master table and how can I contact them?"
  GOOD: "wells_master ownership contact"
  
  BAD: "What is the business purpose of the production_operations schema?"
  GOOD: "production_operations business purpose"
  
  BAD: "How do I access the customer_data table?"
  GOOD: "customer_data access permissions"
  
  TRANSFORMATION RULES:
  1. Extract the KEY ENTITY (table name, schema name, concept)
  2. Extract the INFORMATION TYPE (ownership, contact, purpose, technical details)
  3. Combine as: "[entity] [information type]"
  4. Remove question words (who, what, how, when, where)
  5. Remove filler words (the, a, an, is, are)
  6. Keep 2-5 keywords maximum
  
  EXAMPLES:
  User: "Who owns wells_master table?" → Query: "wells_master ownership contact"
  User: "What's the purpose of financial_analytics?" → Query: "financial_analytics business purpose"
  User: "How to contact the data steward for reservoir_management?" → Query: "reservoir_management steward contact"
  User: "Technical details about pipeline_network table" → Query: "pipeline_network technical details"
  
  Create a step-by-step plan that:
  1. Identifies which tools or agents to use
  2. Determines the optimal sequence of actions
  3. **Transforms queries into keyword format for vector search**
  4. Plans how to synthesize the results into a comprehensive answer
  
  Guidelines:
  - Use vector search tools for specific domain knowledge (identity, business, technical, governance)
  - Use the Genie agent for data analysis, SQL queries, or quantitative questions
  - Keep plans concise (3-5 steps typically) unless the query is very complex
  - Each step should have a clear purpose and expected output
  - The final step should always be "synthesize" to create the user-facing response
  
  Action Types:
  - "tool_call": Execute a specific tool (vector search, UC function, etc.)
  - "genie_query": Query the Genie data analyst agent
  - "synthesize": Combine results into final answer (always the last step)

"""





# COMMAND ----------

# DBTITLE 1,Topic Classifier prompt

topic_classifier_prompt = """You are a topic classifier. Analyze whether the current user query is related to the previous conversation history or represents a completely new topic.
  
  Previous Conversation History:
  {history}
  
  Current User Query:
  {current_query}
  
  Determine:
  1. Is this query a continuation of the previous conversation? (same topic, follow-up question, referring to previous context)
  2. Or is this a completely new, unrelated topic?
  
  Consider:
  - Pronouns referencing previous messages (it, that, them, etc.)
  - Follow-up questions or clarifications
  - Topic continuity and semantic similarity
  - Temporal references (earlier, before, you mentioned, etc.)
  
  Be confident in your classification. If there's any reasonable connection, classify as continuation.
"""


# COMMAND ----------

# DBTITLE 1,Replanner Prompt

replanner_prompt ="""You are an expert replanning agent. A previous execution plan has encountered an error.
  Analyze what went wrong and create a new, improved plan.
  
  Original Plan:
  {original_plan}
  
  Failed at Step: {failed_step}
  
  Error Details:
  {error}
  
  Execution Results So Far:
  {execution_results}
  
  Available Tools:
  {tools}
  
  Create a new plan that:
  1. Addresses the root cause of the failure
  2. Adjusts the approach based on what worked and what didn't
  3. May simplify or add steps as needed
  4. Uses alternative tools or approaches if the original approach is flawed
  
  Common failure reasons and fixes:
  - Tool not found → Use a different available tool
  - Empty results → Broaden search query or try different tool
  - Invalid parameters → Adjust query format or parameters
  - Timeout → Break into smaller steps or use more efficient approach
  
  Ensure the new plan is executable and addresses the user's original query.
"""





# COMMAND ----------

# DBTITLE 1,Register Prompts in MLflow Prompt Registry
# This is for future implementation

import mlflow
from typing import Dict, List, Tuple

mlflow.set_registry_uri("databricks-uc")

PROMPT_CATALOG = "Your Catalog"
PROMPT_SCHEMA = "Your Schema"


def register_prompt(
    prompt_short_name: str,
    prompt_template: str,
    commit_message: str,
    alias: str = "production",
    catalog: str = PROMPT_CATALOG,
    schema: str = PROMPT_SCHEMA
) -> Tuple[str, int]:
    """
    Register a single prompt in Unity Catalog with validation.
    
    Args:
        prompt_short_name: Short name without catalog/schema (e.g., "planner_prompt")
        prompt_template: The actual prompt template text
        commit_message: Description of the prompt or changes
        alias: Alias to set for this version (default: "production")
        catalog: Unity Catalog name
        schema: Unity Catalog schema name
        
    Returns:
        Tuple of (full_prompt_name, version_number)
    """
    # Validate inputs
    if not prompt_template or not prompt_template.strip():
        raise ValueError(f"Cannot register empty prompt: {prompt_short_name}")
    
    # Build full name
    full_name = f"{catalog}.{schema}.{prompt_short_name}"
    
    print(f"\n{'='*80}")
    print(f"Registering: {full_name}")
    print(f"{'='*80}")
    print(f"  Template length: {len(prompt_template)} characters")
    print(f"  First 150 chars: {prompt_template[:150]}...")
    
    try:
        # Register the prompt
        prompt = mlflow.genai.register_prompt(
            name=full_name,
            template=prompt_template,
            commit_message=commit_message
        )
        
        print(f"✅ Created version {prompt.version}")
        
        # Set alias
        mlflow.genai.set_prompt_alias(
            name=full_name,
            alias=alias,
            version=prompt.version
        )
        
        print(f"✅ Set '{alias}' alias to version {prompt.version}")
        
        # Verify it was saved correctly
        # FIX: Use single slash in URI format: prompts:/name@alias
        loaded = mlflow.genai.load_prompt(name_or_uri=f"prompts:/{full_name}@{alias}")
        if not loaded.template or len(loaded.template) < 10:
            raise ValueError(f"Verification failed! Loaded template is empty or too short")
        
        print(f"✅ Verified: loaded template has {len(loaded.template)} characters")
        
        return full_name, prompt.version
        
    except Exception as e:
        print(f"❌ Error registering {full_name}: {e}")
        raise


def register_all_prompts(
    prompts_config: List[Dict[str, str]],
    catalog: str = PROMPT_CATALOG,
    schema: str = PROMPT_SCHEMA,
    alias: str = "production"
) -> Dict[str, Tuple[str, int]]:
    """
    Register multiple prompts iteratively.
    
    Args:
        prompts_config: List of dicts with keys:
            - 'name': short prompt name
            - 'template': prompt template text
            - 'commit_message': description
        catalog: Unity Catalog name
        schema: Unity Catalog schema name
        alias: Alias to set (default: "production")
        
    Returns:
        Dictionary mapping prompt names to (full_name, version) tuples
    """
    results = {}
    errors = []
    
    print(f"\n{'#'*80}")
    print(f"# Registering {len(prompts_config)} prompts")
    print(f"# Catalog: {catalog}.{schema}")
    print(f"# Alias: {alias}")
    print(f"{'#'*80}\n")
    
    for prompt_config in prompts_config:
        try:
            name = prompt_config['name']
            template = prompt_config['template']
            commit_msg = prompt_config['commit_message']
            
            full_name, version = register_prompt(
                prompt_short_name=name,
                prompt_template=template,
                commit_message=commit_msg,
                alias=alias,
                catalog=catalog,
                schema=schema
            )
            
            results[name] = (full_name, version)
            
        except Exception as e:
            error_msg = f"Failed to register '{prompt_config.get('name', 'unknown')}': {e}"
            errors.append(error_msg)
            print(f"\n❌ {error_msg}\n")
    
    # Print summary
    print(f"\n{'#'*80}")
    print(f"# Registration Summary")
    print(f"{'#'*80}")
    print(f"✅ Successfully registered: {len(results)}/{len(prompts_config)} prompts")
    
    if errors:
        print(f"❌ Failed: {len(errors)} prompts")
        for error in errors:
            print(f"   - {error}")
    else:
        print(f"🎉 All prompts registered successfully!")
    
    print(f"{'#'*80}\n")
    
    return results


# Define all prompts to register
prompts_to_register = [
    {
        'name': 'planner_prompt',
        'template': planner_prompt,
        'commit_message': 'Initial planner prompt for multi-agent task planning'
    },
    {
        'name': 'topic_classifier_prompt',
        'template': topic_classifier_prompt,
        'commit_message': 'Initial topic classifier prompt for conversation segmentation'
    },
    {
        'name': 'replanner_prompt',
        'template': replanner_prompt,
        'commit_message': 'Initial replanner prompt for plan refinement'
    },
    {
        'name': 'genie_system_prompt',
        'template': genie_system_prompt,
        'commit_message': 'Initial Genie system prompt for analytics queries'
    },
    {
        'name': 'vector_search_identity_tool_desc',
        'template': vector_search_identity_tool_desc,
        'commit_message': 'Vector search identity tool description'
    },
    {
        'name': 'vector_search_technical_tool_desc',
        'template': vector_search_technical_tool_desc,
        'commit_message': 'Vector search technical tool description'
    },
    {
        'name': 'vector_search_business_tool_desc',
        'template': vector_search_business_tool_desc,
        'commit_message': 'Vector search business tool description'
    },
    {
        'name': 'vector_search_governance_tool_desc',
        'template': vector_search_governance_tool_desc,
        'commit_message': 'Vector search governance tool description'
    }
]

# # Register all prompts
# results = register_all_prompts(
#     prompts_config=prompts_to_register,
#     catalog=PROMPT_CATALOG,
#     schema=PROMPT_SCHEMA,
#     alias="production"
# )

# # Print results for reference
# print("\nRegistered prompts URIs:")
# for name, (full_name, version) in results.items():
#     # FIX: Use single slash in URI format
#     prompt_uri = f"prompts:/{full_name}@production"
#     print(f"  {name}: {prompt_uri} (v{version})")

# COMMAND ----------

# DBTITLE 1,Create Config

import yaml
    
yaml_data = {
    "catalog": "your catalog",
    "schema": "your schema",
    "llm_endpoint": "databricks-claude-3-7-sonnet",
    "prompts": {
        "planner_prompt": planner_prompt,
        "topic_classifier_prompt": topic_classifier_prompt,
        "replanner_prompt": replanner_prompt,
        "genie_system_prompt": genie_system_prompt,
        "vector_search_identity_tool_desc": vector_search_identity_tool_desc,
        "vector_search_technical_tool_desc": vector_search_technical_tool_desc,
        "vector_search_business_tool_desc": vector_search_business_tool_desc,
        "vector_search_governance_tool_desc": vector_search_governance_tool_desc,
        "synthesis_prompt":synthesis_prompt
    },
    "agent_name": "agent",
    "tools" : {
                "vector_search_identity_index":"yourcatalog.yourschema.metadata_identity_idx",
                "vector_search_technical_index":"yourcatalog.yourschema.metadata_technical_idx",
                "vector_search_business_index":"yourcatalog.yourschema.metadata_business_idx",
                "vector_search_governance_index":"yourcatalog.yourschema.metadata_governance_idx",
                "vector_search_embedding_endpoint":"databricks-bge-large-en",
                "genie_space_id":"01f09d571a1e1bf4bcbdc1c562fd1167",
                "genie_agent_name" : "Genie",
                "genie_table_schema": "data_discovery",
                },
    "lakebase_config":{
                "instance_name" : "your instance name"  ,
                "conn_db_name":"your db name",
                "conn_ssl_mode":"require",
                "conn_host":"instance-XXXXX.database.azuredatabricks.net",
                "conn_port":5432,
                "workspace_host":"https://adb-XXXXXXXX.XX.azuredatabricks.net",
                "recursion_limit":25,}
}

with open("config.yaml", "w") as f:
    yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

# COMMAND ----------

# DBTITLE 1,Verify Lakebase connectivity
import os
import uuid
from databricks.sdk import WorkspaceClient
from psycopg_pool import ConnectionPool
import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from mlflow.models import ModelConfig

config = ModelConfig(development_config="config.yaml").to_dict()


lakebase_port = config['lakebase_config']["conn_port"]
lakebase_ssl_mode = config['lakebase_config']["conn_ssl_mode"] 
database_name=  config['lakebase_config']['conn_db_name']
lakebase_workspace_host=config['lakebase_config']['workspace_host']
lakebase_host=config['lakebase_config']['conn_host']
lakebase_instance_name=config['lakebase_config']['instance_name']

w = WorkspaceClient()
def db_password_provider() -> str:
    """
    Ask Databricks to mint a fresh DB credential for this instance.
    """
    try:
        cred = w.database.generate_database_credential(
            request_id=str(uuid.uuid4()),
            instance_names=[lakebase_instance_name]
        )
        return cred.token
    except Exception as e:
        print(f"Error generating database credential: {e}")
        return

class CredentialConnection(psycopg.Connection):
    """
    A psycopg Connection subclass that injects a fresh password
    *at connection time* (only when the pool creates a new connection).
    """
    @classmethod
    def connect(cls, conninfo="", **kwargs):
        # Append the new password to kwargs
        kwargs["password"] = db_password_provider() 
        # Call the superclass's connect method with updated kwargs
        return super().connect(conninfo, **kwargs)



pool = ConnectionPool(
            conninfo=f"dbname={database_name} user=yourusername@company.com host={lakebase_host} port={lakebase_port} sslmode={lakebase_ssl_mode}",
            connection_class=CredentialConnection,
            open=True,
            kwargs={
                "autocommit": True,  # Required for .setup() method
                "keepalives": 1,
                "keepalives_idle": 30,
                "keepalives_interval": 10,
                "keepalives_count": 5,
            }
        )



# Use the pool to initialize your checkpoint tables
with pool.connection() as conn:
    conn.autocommit = True   # disable transaction wrapping

    with conn.cursor() as cur:
        cur.execute("select version();")
    print("✅ Pool connected and checkpoint tables are ready.")

# COMMAND ----------

# DBTITLE 1,Create Message Converters
# MAGIC %%writefile message_converters.py
# MAGIC
# MAGIC """
# MAGIC Streamlined message conversion utilities for MLflow ResponsesAgent integration.
# MAGIC
# MAGIC This module provides minimal, efficient message normalization focusing on:
# MAGIC 1. Checkpoint deserialization handling (PostgreSQL state management)
# MAGIC 2. Content structure normalization (list/dict to string conversion)
# MAGIC 3. LangChain <-> ResponsesAgent interoperability
# MAGIC
# MAGIC NOTE: Modern databricks-langchain and MLflow 3.x handle most conversions natively.
# MAGIC This utility focuses only on edge cases and checkpoint-specific issues.
# MAGIC """
# MAGIC
# MAGIC import logging
# MAGIC from typing import Any, Dict, List, Union
# MAGIC
# MAGIC  
# MAGIC
# MAGIC from langchain_core.messages import (
# MAGIC     BaseMessage,
# MAGIC     HumanMessage,
# MAGIC     AIMessage,
# MAGIC     SystemMessage,
# MAGIC     ToolMessage,
# MAGIC )
# MAGIC
# MAGIC logger = logging.getLogger(__name__)
# MAGIC
# MAGIC
# MAGIC class MessageNormalizer:
# MAGIC     """
# MAGIC     Lightweight message normalization for checkpoint deserialization.
# MAGIC     
# MAGIC     Primary purpose: Handle complex content structures that emerge from
# MAGIC     PostgreSQL checkpoint serialization/deserialization cycles.
# MAGIC     """
# MAGIC     
# MAGIC     @staticmethod
# MAGIC     def normalize_content(content: Any) -> str:
# MAGIC         """
# MAGIC         Normalize message content to plain string format.
# MAGIC         
# MAGIC         PostgreSQL checkpoints can serialize content as:
# MAGIC         - Plain strings (ideal case)
# MAGIC         - Lists of dicts: [{"type": "text", "text": "..."}]
# MAGIC         - Complex nested structures from tool outputs
# MAGIC         
# MAGIC         This ensures all content becomes a simple string suitable for LLM input.
# MAGIC         
# MAGIC         Args:
# MAGIC             content: Message content (string, list, dict, or other)
# MAGIC             
# MAGIC         Returns:
# MAGIC             Normalized string content
# MAGIC             
# MAGIC         Examples:
# MAGIC             >>> normalize_content("Hello")
# MAGIC             "Hello"
# MAGIC             
# MAGIC             >>> normalize_content([{"type": "text", "text": "Hello"}])
# MAGIC             "Hello"
# MAGIC             
# MAGIC             >>> normalize_content([{"type": "text", "text": "A"}, {"type": "text", "text": "B"}])
# MAGIC             "A B"
# MAGIC         """
# MAGIC         # Fast path: already a string
# MAGIC         if isinstance(content, str):
# MAGIC             return content
# MAGIC         
# MAGIC         # Handle None/empty
# MAGIC         if not content:
# MAGIC             return ""
# MAGIC         
# MAGIC         # Handle list content (common from checkpoints)
# MAGIC         if isinstance(content, list):
# MAGIC             parts = []
# MAGIC             for item in content:
# MAGIC                 if isinstance(item, dict):
# MAGIC                     # Priority: 'text' field (most common)
# MAGIC                     if "text" in item:
# MAGIC                         parts.append(str(item["text"]))
# MAGIC                     # Fallback: 'content' field
# MAGIC                     elif "content" in item:
# MAGIC                         parts.append(str(item["content"]))
# MAGIC                     # Last resort: stringify entire dict
# MAGIC                     else:
# MAGIC                         parts.append(str(item))
# MAGIC                 elif isinstance(item, str):
# MAGIC                     parts.append(item)
# MAGIC                 else:
# MAGIC                     parts.append(str(item))
# MAGIC             
# MAGIC             return " ".join(parts).strip()
# MAGIC         
# MAGIC         # Handle dict content
# MAGIC         if isinstance(content, dict):
# MAGIC             if "text" in content:
# MAGIC                 return str(content["text"])
# MAGIC             elif "content" in content:
# MAGIC                 return str(content["content"])
# MAGIC             else:
# MAGIC                 return str(content)
# MAGIC         
# MAGIC         # Ultimate fallback
# MAGIC         return str(content)
# MAGIC     
# MAGIC     @staticmethod
# MAGIC     def normalize_message(msg: Union[BaseMessage, Dict]) -> BaseMessage:
# MAGIC         """
# MAGIC         Normalize a single message to proper LangChain BaseMessage format.
# MAGIC         
# MAGIC         Handles:
# MAGIC         - BaseMessage objects with complex content structures
# MAGIC         - Dict representations of messages
# MAGIC         - Content normalization for checkpoint compatibility
# MAGIC         
# MAGIC         Args:
# MAGIC             msg: Message as BaseMessage or dict
# MAGIC             
# MAGIC         Returns:
# MAGIC             Properly formatted BaseMessage with normalized content
# MAGIC         """
# MAGIC         # Handle BaseMessage objects
# MAGIC         if isinstance(msg, BaseMessage):
# MAGIC             # Normalize content structure
# MAGIC             normalized_content = MessageNormalizer.normalize_content(msg.content)
# MAGIC             
# MAGIC             # Recreate message with normalized content
# MAGIC             if msg.type == "human":
# MAGIC                 return HumanMessage(content=normalized_content)
# MAGIC             elif msg.type == "ai":
# MAGIC                 return AIMessage(
# MAGIC                     content=normalized_content,
# MAGIC                     tool_calls=getattr(msg, 'tool_calls', [])
# MAGIC                 )
# MAGIC             elif msg.type == "system":
# MAGIC                 return SystemMessage(content=normalized_content)
# MAGIC             elif msg.type == "tool":
# MAGIC                 return ToolMessage(
# MAGIC                     content=normalized_content,
# MAGIC                     tool_call_id=getattr(msg, 'tool_call_id', '')
# MAGIC                 )
# MAGIC             else:
# MAGIC                 # Unknown type, default to AIMessage
# MAGIC                 logger.warning(f"Unknown message type: {msg.type}, defaulting to AIMessage")
# MAGIC                 return AIMessage(content=normalized_content)
# MAGIC         
# MAGIC         # Handle dict representations
# MAGIC         elif isinstance(msg, dict):
# MAGIC             role = msg.get("role", msg.get("type", "user"))
# MAGIC             content = MessageNormalizer.normalize_content(msg.get("content", ""))
# MAGIC             
# MAGIC             if role in ("user", "human"):
# MAGIC                 return HumanMessage(content=content)
# MAGIC             elif role in ("assistant", "ai"):
# MAGIC                 return AIMessage(
# MAGIC                     content=content,
# MAGIC                     tool_calls=msg.get("tool_calls", [])
# MAGIC                 )
# MAGIC             elif role == "system":
# MAGIC                 return SystemMessage(content=content)
# MAGIC             elif role == "tool":
# MAGIC                 return ToolMessage(
# MAGIC                     content=content,
# MAGIC                     tool_call_id=msg.get("tool_call_id", "")
# MAGIC                 )
# MAGIC             else:
# MAGIC                 logger.warning(f"Unknown role: {role}, defaulting to HumanMessage")
# MAGIC                 return HumanMessage(content=content)
# MAGIC         
# MAGIC         # Fallback for unexpected types
# MAGIC         else:
# MAGIC             logger.warning(f"Unexpected message type: {type(msg)}, converting to string")
# MAGIC             return HumanMessage(content=str(msg))
# MAGIC     
# MAGIC     @staticmethod
# MAGIC     def normalize_messages(messages: List[Union[BaseMessage, Dict]]) -> List[BaseMessage]:
# MAGIC         """
# MAGIC         Normalize a list of messages for LLM consumption.
# MAGIC         
# MAGIC         Critical for:
# MAGIC         - Loading conversation history from PostgreSQL checkpoints
# MAGIC         - Ensuring content compatibility with LLM providers (OpenAI, etc.)
# MAGIC         - Handling mixed message formats
# MAGIC         
# MAGIC         Args:
# MAGIC             messages: List of messages (BaseMessage objects or dicts)
# MAGIC             
# MAGIC         Returns:
# MAGIC             List of properly formatted BaseMessage objects
# MAGIC         """
# MAGIC         return [MessageNormalizer.normalize_message(msg) for msg in messages]
# MAGIC     
# MAGIC     @staticmethod
# MAGIC     def filter_empty_messages(messages: List[BaseMessage]) -> List[BaseMessage]:
# MAGIC         """
# MAGIC         Remove messages with empty content (except those with tool_calls or tool responses).
# MAGIC         
# MAGIC         Prevents errors like: "Invalid 'messages[x].content': string expected, got empty list"
# MAGIC         
# MAGIC         Args:
# MAGIC             messages: List of BaseMessage objects
# MAGIC             
# MAGIC         Returns:
# MAGIC             Filtered list with only valid messages
# MAGIC         """
# MAGIC         filtered = []
# MAGIC         for msg in messages:
# MAGIC             # Always include system messages
# MAGIC             if isinstance(msg, SystemMessage):
# MAGIC                 filtered.append(msg)
# MAGIC             # AI messages are valid if they have content OR tool_calls
# MAGIC             elif isinstance(msg, AIMessage):
# MAGIC                 if msg.content or msg.tool_calls:
# MAGIC                     filtered.append(msg)
# MAGIC             # Tool messages are valid if they have content
# MAGIC             elif isinstance(msg, ToolMessage):
# MAGIC                 if msg.content:
# MAGIC                     filtered.append(msg)
# MAGIC             # Other message types need non-empty content
# MAGIC             elif hasattr(msg, 'content') and msg.content:
# MAGIC                 filtered.append(msg)
# MAGIC         
# MAGIC         return filtered
# MAGIC
# MAGIC
# MAGIC class ResponsesConverter:
# MAGIC     """
# MAGIC     Utilities for converting between MLflow ResponsesAgent format and LangChain.
# MAGIC     
# MAGIC     NOTE: MLflow 3.x and databricks-langchain handle most conversions automatically.
# MAGIC     These are helper methods for explicit conversion when needed.
# MAGIC     """
# MAGIC     
# MAGIC     @staticmethod
# MAGIC     def responses_to_langchain(input_items: List[Any]) -> List[BaseMessage]:
# MAGIC         """
# MAGIC         Convert MLflow Responses API input items to LangChain messages.
# MAGIC         
# MAGIC         Args:
# MAGIC             input_items: List of ResponsesAgentRequestItem objects
# MAGIC             
# MAGIC         Returns:
# MAGIC             List of LangChain BaseMessage objects
# MAGIC         """
# MAGIC         messages = []
# MAGIC         
# MAGIC         for item in input_items:
# MAGIC             try:
# MAGIC                 # Convert to dict
# MAGIC                 if hasattr(item, 'model_dump'):
# MAGIC                     item_dict = item.model_dump()
# MAGIC                 elif hasattr(item, 'dict'):
# MAGIC                     item_dict = item.dict()
# MAGIC                 elif isinstance(item, dict):
# MAGIC                     item_dict = item
# MAGIC                 else:
# MAGIC                     # Fallback: treat as user message
# MAGIC                     messages.append(HumanMessage(content=str(item)))
# MAGIC                     continue
# MAGIC                 
# MAGIC                 role = item_dict.get("role", "user")
# MAGIC                 content = item_dict.get("content", "")
# MAGIC                 
# MAGIC                 if role == "user":
# MAGIC                     messages.append(HumanMessage(content=content))
# MAGIC                 elif role == "assistant":
# MAGIC                     messages.append(AIMessage(
# MAGIC                         content=content,
# MAGIC                         tool_calls=item_dict.get("tool_calls", [])
# MAGIC                     ))
# MAGIC                 elif role == "system":
# MAGIC                     messages.append(SystemMessage(content=content))
# MAGIC                 elif role == "tool":
# MAGIC                     messages.append(ToolMessage(
# MAGIC                         content=content,
# MAGIC                         tool_call_id=item_dict.get("tool_call_id", "")
# MAGIC                     ))
# MAGIC                 else:
# MAGIC                     # Unknown role, default to user message
# MAGIC                     messages.append(HumanMessage(content=content))
# MAGIC                     
# MAGIC             except Exception as e:
# MAGIC                 logger.error(f"Error converting input item: {e}")
# MAGIC                 # Fallback: create user message
# MAGIC                 messages.append(HumanMessage(content=str(item)))
# MAGIC         
# MAGIC         return messages
# MAGIC     
# MAGIC     @staticmethod
# MAGIC     def langchain_to_responses_dict(message: BaseMessage) -> Dict[str, Any]:
# MAGIC         """
# MAGIC         Convert a LangChain message to ResponsesAgent dict format.
# MAGIC         
# MAGIC         Useful for manual response construction or debugging.
# MAGIC         
# MAGIC         Args:
# MAGIC             message: LangChain BaseMessage
# MAGIC             
# MAGIC         Returns:
# MAGIC             Dict in ResponsesAgent format
# MAGIC         """
# MAGIC         message_dict = message.model_dump()
# MAGIC         role = message_dict["type"]
# MAGIC         
# MAGIC         result = {
# MAGIC             "role": role,
# MAGIC             "content": MessageNormalizer.normalize_content(message_dict.get("content", ""))
# MAGIC         }
# MAGIC         
# MAGIC         # Add tool-specific fields
# MAGIC         if role == "ai" and message_dict.get("tool_calls"):
# MAGIC             result["tool_calls"] = message_dict["tool_calls"]
# MAGIC         elif role == "tool":
# MAGIC             result["tool_call_id"] = message_dict.get("tool_call_id", "")
# MAGIC         
# MAGIC         return result
# MAGIC
# MAGIC
# MAGIC
# MAGIC def normalize_checkpoint_messages(messages: List[Union[BaseMessage, Dict]]) -> List[BaseMessage]:
# MAGIC     """
# MAGIC     One-stop normalization for messages loaded from checkpoints.
# MAGIC     
# MAGIC     Applies both content normalization and empty message filtering.
# MAGIC     Use this when loading conversation history from PostgreSQL state.
# MAGIC     
# MAGIC     Args:
# MAGIC         messages: Raw messages from checkpoint
# MAGIC         
# MAGIC     Returns:
# MAGIC         Clean, normalized messages ready for LLM
# MAGIC     """
# MAGIC     normalized = MessageNormalizer.normalize_messages(messages)
# MAGIC     return MessageNormalizer.filter_empty_messages(normalized)
# MAGIC

# COMMAND ----------

# DBTITLE 1,Initialise Checkpoint Tables if they dont exist
# # Use only one when initialising
# from lb_state_manager import DatabricksStateManager
# from mlflow.models import ModelConfig

# config = ModelConfig(development_config="config.yaml").to_dict()
# state_manager = DatabricksStateManager(lakebase_config=config["lakebase_config"])

# # Initialize tables
# state_manager.initialize_checkpoint_tables()

# COMMAND ----------

# DBTITLE 1,agent.py
# MAGIC %%writefile agentv1_7.py
# MAGIC import functools
# MAGIC import json
# MAGIC import logging
# MAGIC import os
# MAGIC import uuid
# MAGIC from typing import Any, Generator, Literal, Optional, Dict, List, Union, Sequence, Annotated, TypedDict
# MAGIC from enum import Enum
# MAGIC import mlflow
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks_langchain import (
# MAGIC     ChatDatabricks,
# MAGIC     UCFunctionToolkit,
# MAGIC     VectorSearchRetrieverTool,
# MAGIC )
# MAGIC from langchain_core.language_models import LanguageModelLike
# MAGIC from langchain_core.tools import BaseTool 
# MAGIC from langchain_core.messages import (
# MAGIC     AIMessage,
# MAGIC     AIMessageChunk,
# MAGIC     BaseMessage,
# MAGIC     SystemMessage,
# MAGIC     HumanMessage,
# MAGIC     ToolMessage,
# MAGIC )
# MAGIC from databricks_langchain.genie import GenieAgent
# MAGIC from langchain_core.runnables import RunnableLambda, RunnableConfig
# MAGIC from langgraph.graph import END, StateGraph
# MAGIC from langgraph.graph.message import add_messages
# MAGIC from langgraph.prebuilt import ToolNode
# MAGIC from langgraph.checkpoint.postgres import PostgresSaver
# MAGIC from mlflow.pyfunc import ResponsesAgent
# MAGIC from mlflow.types.responses import (
# MAGIC     ResponsesAgentRequest,
# MAGIC     ResponsesAgentResponse,
# MAGIC     ResponsesAgentStreamEvent,
# MAGIC )
# MAGIC from mlflow.models import ModelConfig
# MAGIC from pydantic import BaseModel, Field
# MAGIC
# MAGIC # Import utilities 
# MAGIC from lb_state_manager import DatabricksStateManager
# MAGIC from message_converters import MessageNormalizer
# MAGIC from functools import wraps
# MAGIC
# MAGIC logger = logging.getLogger(__name__)
# MAGIC logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
# MAGIC
# MAGIC config = ModelConfig(development_config="config.yaml").to_dict()
# MAGIC mlflow.set_tracking_uri("databricks")
# MAGIC
# MAGIC ###################################################
# MAGIC ## Configuration
# MAGIC ###################################################
# MAGIC
# MAGIC LLM_ENDPOINT_NAME = config['llm_endpoint']
# MAGIC GENIE_SPACE_ID = config['tools']['genie_space_id'] 
# MAGIC GENIE_AGENT_NAME = config['tools']['genie_agent_name']
# MAGIC GENIE_AGENT_DESCRIPTION = config['prompts']['genie_system_prompt']
# MAGIC # TOOLS_SYSTEM_PROMPT = config['prompts']['tools_system_prompt']
# MAGIC
# MAGIC # Prompts for planner architecture
# MAGIC TOPIC_CLASSIFIER_PROMPT = config['prompts']['topic_classifier_prompt']
# MAGIC PLANNER_PROMPT = config['prompts']['planner_prompt']
# MAGIC REPLANNER_PROMPT = config['prompts']['replanner_prompt']
# MAGIC
# MAGIC ###################################################
# MAGIC ## Initialize LLM and Tools
# MAGIC ###################################################
# MAGIC
# MAGIC llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)
# MAGIC
# MAGIC # Initialize tools
# MAGIC tools = []
# MAGIC UC_TOOL_NAMES: list[str] = [] 
# MAGIC if UC_TOOL_NAMES:
# MAGIC     uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
# MAGIC     tools.extend(uc_toolkit.tools)
# MAGIC
# MAGIC # Vector search tools
# MAGIC VECTOR_SEARCH_TOOLS = [
# MAGIC     VectorSearchRetrieverTool(
# MAGIC         index_name=config['tools']["vector_search_identity_index"],
# MAGIC         tool_name="identity_search",
# MAGIC         tool_description=config["prompts"]["vector_search_identity_tool_desc"]
# MAGIC     ),
# MAGIC     VectorSearchRetrieverTool(
# MAGIC         index_name=config["tools"]["vector_search_business_index"],
# MAGIC         tool_name="business_search",
# MAGIC         tool_description=config["prompts"]["vector_search_business_tool_desc"]
# MAGIC     ),
# MAGIC     VectorSearchRetrieverTool(
# MAGIC         index_name=config["tools"]["vector_search_technical_index"],
# MAGIC         tool_name="technical_search",
# MAGIC         tool_description=config["prompts"]["vector_search_technical_tool_desc"]
# MAGIC     ),
# MAGIC     VectorSearchRetrieverTool(
# MAGIC         index_name=config["tools"]["vector_search_governance_index"],
# MAGIC         tool_name="governance_search",
# MAGIC         tool_description=config["prompts"]["vector_search_governance_tool_desc"]
# MAGIC     ),
# MAGIC ]
# MAGIC
# MAGIC tools.extend(VECTOR_SEARCH_TOOLS)
# MAGIC
# MAGIC # Genie Agent
# MAGIC genie_agent = GenieAgent(
# MAGIC     genie_space_id=GENIE_SPACE_ID,
# MAGIC     genie_agent_name=GENIE_AGENT_NAME,
# MAGIC     description=GENIE_AGENT_DESCRIPTION
# MAGIC )
# MAGIC
# MAGIC ###################################################
# MAGIC ## Pydantic Models for Planning
# MAGIC ###################################################
# MAGIC
# MAGIC class PlanStep(BaseModel):
# MAGIC     """A single step in the execution plan."""
# MAGIC     step_number: int = Field(description="Sequential step number")
# MAGIC     action: Literal["tool_call", "genie_query", "synthesize"] = Field(
# MAGIC         description="Type of action to perform"
# MAGIC     )
# MAGIC     tool_name: Optional[str] = Field(
# MAGIC         default=None, 
# MAGIC         description="Name of tool to call (for tool_call action)"
# MAGIC     )
# MAGIC     query: Optional[str] = Field(
# MAGIC         default=None,
# MAGIC         description="Query or input for the action"
# MAGIC     )
# MAGIC     reasoning: str = Field(description="Why this step is needed")
# MAGIC
# MAGIC
# MAGIC class Plan(BaseModel):
# MAGIC     """Complete execution plan."""
# MAGIC     steps: List[PlanStep] = Field(description="Ordered list of steps to execute")
# MAGIC     overall_strategy: str = Field(description="High-level strategy for answering the query")
# MAGIC
# MAGIC
# MAGIC class Classification(BaseModel):
# MAGIC     """Classification result for query context."""
# MAGIC     is_new_topic: bool = Field(description="True if query is unrelated to conversation history")
# MAGIC     reasoning: str = Field(description="Explanation for the classification")
# MAGIC     confidence: float = Field(description="Confidence score between 0 and 1")
# MAGIC
# MAGIC class StateNormalizer:
# MAGIC     """Handles state object reconstruction from PostgreSQL checkpoints."""
# MAGIC     
# MAGIC     @staticmethod
# MAGIC     def normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
# MAGIC         """
# MAGIC         Reconstruct Pydantic objects from checkpoint dicts.
# MAGIC         
# MAGIC         PostgreSQL checkpointing serializes Pydantic objects as dicts.
# MAGIC         This reconstructs them back to proper Pydantic objects for type safety.
# MAGIC         
# MAGIC         Args:
# MAGIC             state: AgentState dict (may contain serialized Pydantic objects)
# MAGIC             
# MAGIC         Returns:
# MAGIC             State with Pydantic objects reconstructed
# MAGIC         """
# MAGIC         
# MAGIC         # Reconstruct Classification object
# MAGIC         if "classification" in state and state["classification"] is not None:
# MAGIC             if isinstance(state["classification"], dict):
# MAGIC                 state["classification"] = Classification(**state["classification"])
# MAGIC         
# MAGIC         # Reconstruct Plan object (with nested PlanStep objects)
# MAGIC         if "plan" in state and state["plan"] is not None:
# MAGIC             if isinstance(state["plan"], dict):
# MAGIC                 plan_dict = state["plan"].copy()
# MAGIC                 
# MAGIC                 # Reconstruct nested PlanStep objects
# MAGIC                 if "steps" in plan_dict and plan_dict["steps"]:
# MAGIC                     plan_dict["steps"] = [
# MAGIC                         PlanStep(**step) if isinstance(step, dict) else step
# MAGIC                         for step in plan_dict["steps"]
# MAGIC                     ]
# MAGIC                 
# MAGIC                 state["plan"] = Plan(**plan_dict)
# MAGIC         
# MAGIC         return state
# MAGIC # Convenience function for the most common use case
# MAGIC
# MAGIC def normalize_state_inputs(func):
# MAGIC     """
# MAGIC     Decorator to normalize state before node execution.
# MAGIC     
# MAGIC     Ensures Pydantic objects are reconstructed from checkpoint dicts.
# MAGIC     Apply to all LangGraph node functions that read state.
# MAGIC     
# MAGIC     Usage:
# MAGIC         @normalize_state_inputs
# MAGIC         def my_node(state: AgentState) -> AgentState:
# MAGIC             # State is already normalized here
# MAGIC             ...
# MAGIC     """
# MAGIC     @wraps(func)
# MAGIC     def wrapper(state):
# MAGIC         state = StateNormalizer.normalize_state(state)
# MAGIC         return func(state)
# MAGIC     return wrapper
# MAGIC
# MAGIC
# MAGIC ###################################################
# MAGIC ## State Schema
# MAGIC ###################################################
# MAGIC
# MAGIC class AgentState(TypedDict):
# MAGIC     """State for the planner-based agent."""
# MAGIC     messages: Annotated[List[BaseMessage], add_messages]
# MAGIC     classification: Optional[Classification]
# MAGIC     context_messages: List[BaseMessage]  # Filtered messages for planning
# MAGIC     plan: Optional[Plan]
# MAGIC     current_step: int
# MAGIC     execution_results: Dict[int, Any]  # step_number -> result
# MAGIC     replan_count: int
# MAGIC     final_response: Optional[str]
# MAGIC     error: Optional[str]
# MAGIC
# MAGIC
# MAGIC ###################################################
# MAGIC ## Helper Functions
# MAGIC ###################################################
# MAGIC
# MAGIC def build_tool_descriptions() -> str:
# MAGIC     """Build formatted descriptions of all available tools."""
# MAGIC     descriptions = []
# MAGIC     
# MAGIC     # Vector search tools
# MAGIC     for tool in VECTOR_SEARCH_TOOLS:
# MAGIC         descriptions.append(f"- {tool.name}: {tool.description}")
# MAGIC     
# MAGIC     # UC tools
# MAGIC     for tool in tools:
# MAGIC         if tool.name not in [t.name for t in VECTOR_SEARCH_TOOLS]:
# MAGIC             descriptions.append(f"- {tool.name}: {tool.description}")
# MAGIC     
# MAGIC     # Genie agent
# MAGIC     descriptions.append(f"- {GENIE_AGENT_NAME}: {GENIE_AGENT_DESCRIPTION}")
# MAGIC     
# MAGIC     return "\n".join(descriptions)
# MAGIC
# MAGIC
# MAGIC ###################################################
# MAGIC ## Node Functions
# MAGIC ###################################################
# MAGIC @normalize_state_inputs
# MAGIC def classify_topic(state: AgentState) -> AgentState:
# MAGIC     """Classify if the current query is a new topic or continuation."""
# MAGIC     messages = state.get("messages", [])
# MAGIC     
# MAGIC     # Normalize messages from checkpoint (handle complex content structures)
# MAGIC     messages = MessageNormalizer.normalize_messages(messages)
# MAGIC     state["messages"] = messages
# MAGIC     
# MAGIC     # If no history, it's definitely a new topic
# MAGIC     if len(messages) <= 1:
# MAGIC         state["classification"] = Classification(
# MAGIC             is_new_topic=True,
# MAGIC             reasoning="No conversation history exists",
# MAGIC             confidence=1.0
# MAGIC         )
# MAGIC         state["context_messages"] = [messages[-1]] if messages else []
# MAGIC         return state
# MAGIC     
# MAGIC     # Use LLM to classify
# MAGIC     current_query = messages[-1].content if messages[-1].content else ""
# MAGIC     
# MAGIC     # Build history summary from recent messages
# MAGIC     history_messages = messages[-6:-1] if len(messages) > 1 else []
# MAGIC     history_summary = "\n".join([
# MAGIC         f"{msg.type}: {msg.content[:200]}..." if len(msg.content) > 200 else f"{msg.type}: {msg.content}"
# MAGIC         for msg in history_messages
# MAGIC         if hasattr(msg, 'content') and msg.content
# MAGIC     ])
# MAGIC     
# MAGIC     if not history_summary:
# MAGIC         history_summary = "No previous conversation"
# MAGIC     
# MAGIC     try:
# MAGIC         classification_prompt = TOPIC_CLASSIFIER_PROMPT.format(
# MAGIC             history=history_summary,
# MAGIC             current_query=current_query
# MAGIC         )
# MAGIC         
# MAGIC         structured_llm = llm.with_structured_output(Classification)
# MAGIC         classification = structured_llm.invoke([HumanMessage(content=classification_prompt)])
# MAGIC         
# MAGIC         state["classification"] = classification
# MAGIC         
# MAGIC         # Prepare context messages
# MAGIC         if classification.is_new_topic:
# MAGIC             # Only use current query
# MAGIC             state["context_messages"] = [messages[-1]]
# MAGIC             logger.info(f"Classified as NEW TOPIC (confidence: {classification.confidence})")
# MAGIC         else:
# MAGIC             # Include relevant history (last 10 messages)
# MAGIC             state["context_messages"] = messages[-10:]
# MAGIC             logger.info(f"Classified as CONTINUATION (confidence: {classification.confidence})")
# MAGIC     except Exception as e:
# MAGIC         logger.error(f"Classification failed: {str(e)}", exc_info=True)
# MAGIC         # Default to new topic on error
# MAGIC         state["classification"] = Classification(
# MAGIC             is_new_topic=True,
# MAGIC             reasoning=f"Classification failed: {str(e)}",
# MAGIC             confidence=0.5
# MAGIC         )
# MAGIC         state["context_messages"] = [messages[-1]] if messages else []
# MAGIC     
# MAGIC     return state
# MAGIC
# MAGIC @normalize_state_inputs
# MAGIC def create_plan(state: AgentState) -> AgentState:
# MAGIC     """Generate execution plan based on query and context."""
# MAGIC     context_messages = state.get("context_messages", [])
# MAGIC     if not context_messages:
# MAGIC         logger.error("No context messages available for planning")
# MAGIC         state["error"] = "No context messages available"
# MAGIC         return state
# MAGIC     
# MAGIC     current_query = context_messages[-1].content
# MAGIC
# MAGIC     logger.info(f"=== PLANNING DEBUG ===")
# MAGIC     logger.info(f"Current query: {current_query}")
# MAGIC     logger.info(f"Context messages count: {len(context_messages)}")
# MAGIC
# MAGIC     # Build context with tool descriptions
# MAGIC     tool_descriptions = build_tool_descriptions()
# MAGIC     logger.info(f"Tool descriptions length: {len(tool_descriptions)}")
# MAGIC     logger.info(f"Tool descriptions preview: {tool_descriptions[:200]}...")
# MAGIC
# MAGIC     # Build context string
# MAGIC     context_str = "\n".join([
# MAGIC         f"{msg.type}: {msg.content}" 
# MAGIC         for msg in context_messages[:-1]
# MAGIC         if hasattr(msg, 'content') and msg.content
# MAGIC     ]) if len(context_messages) > 1 else "No previous context"
# MAGIC     
# MAGIC     logger.info(f"Context string: {context_str}")
# MAGIC
# MAGIC     try:
# MAGIC         planning_prompt = PLANNER_PROMPT.format(
# MAGIC             tools=tool_descriptions,
# MAGIC             query=current_query,
# MAGIC             context=context_str
# MAGIC         )
# MAGIC         logger.info(f"Planning prompt length: {len(planning_prompt)}")
# MAGIC         logger.info(f"Planning prompt first 300 chars: {planning_prompt[:300]}")
# MAGIC     except KeyError as e:
# MAGIC         logger.error(f"Prompt formatting failed - missing key: {e}")
# MAGIC         state["error"] = f"Prompt formatting failed: {e}"
# MAGIC         return state
# MAGIC     
# MAGIC     try:
# MAGIC         if not planning_prompt or not planning_prompt.strip():
# MAGIC             logger.error(f"Planning prompt is empty! Tool descriptions length: {len(tool_descriptions)}, Query: {current_query}")
# MAGIC             state["error"] = "Planning prompt is empty"
# MAGIC             return state
# MAGIC
# MAGIC         logger.debug(f"Planning prompt length: {len(planning_prompt)}")
# MAGIC         logger.debug(f"Planning prompt preview: {planning_prompt[:200]}...")
# MAGIC
# MAGIC         structured_llm = llm.with_structured_output(Plan)
# MAGIC         plan = structured_llm.invoke([HumanMessage(content=planning_prompt)])
# MAGIC         
# MAGIC         state["plan"] = plan
# MAGIC         state["current_step"] = 0
# MAGIC         state["execution_results"] = {}
# MAGIC         state["replan_count"] = state.get("replan_count", 0)  # Initialize if not present
# MAGIC         
# MAGIC         logger.info(f"Generated plan with {len(plan.steps)} steps")
# MAGIC         logger.info(f"Strategy: {plan.overall_strategy}")
# MAGIC     except Exception as e:
# MAGIC         logger.error(f"Failed to generate plan: {str(e)}", exc_info=True)
# MAGIC         state["error"] = f"Planning failed: {str(e)}"
# MAGIC     
# MAGIC     return state
# MAGIC
# MAGIC @normalize_state_inputs
# MAGIC def replan(state: AgentState) -> AgentState:
# MAGIC     """Replan based on execution failures."""
# MAGIC     state["replan_count"] = state.get("replan_count", 0) + 1
# MAGIC     
# MAGIC     if state["replan_count"] > 2:
# MAGIC         state["error"] = "Maximum replan attempts (2) exceeded"
# MAGIC         return state
# MAGIC     
# MAGIC     # Get context about what failed
# MAGIC     failed_step = 0
# MAGIC     if "current_step" in state:
# MAGIC         failed_step = state["current_step"]
# MAGIC     original_plan = state.get("plan")
# MAGIC     if not original_plan:
# MAGIC         logger.error("No plan available for replanning")
# MAGIC         state["error"] = "Cannot replan without an original plan"
# MAGIC         return state
# MAGIC     execution_results = state["execution_results"]
# MAGIC     error_info = state.get("error", "Unknown error")
# MAGIC     
# MAGIC     replanning_prompt = REPLANNER_PROMPT.format(
# MAGIC         original_plan=json.dumps(original_plan.model_dump(), indent=2),
# MAGIC         failed_step=failed_step,
# MAGIC         error=error_info,
# MAGIC         execution_results=json.dumps(execution_results, indent=2),
# MAGIC         tools=build_tool_descriptions()
# MAGIC     )
# MAGIC     
# MAGIC     structured_llm = llm.with_structured_output(Plan)
# MAGIC     new_plan = structured_llm.invoke([HumanMessage(content=replanning_prompt)])
# MAGIC     
# MAGIC     state["plan"] = new_plan
# MAGIC     state["current_step"] = 0
# MAGIC     state["error"] = None
# MAGIC     
# MAGIC     logger.info(f"Replanned (attempt {state['replan_count']}): {new_plan.overall_strategy}")
# MAGIC     
# MAGIC     return state
# MAGIC
# MAGIC @normalize_state_inputs
# MAGIC def execute_step(state: AgentState) -> AgentState:
# MAGIC     """Execute a single step from the plan."""
# MAGIC     plan = state.get("plan")
# MAGIC     if not plan:
# MAGIC         logger.error("No plan found in state")
# MAGIC         state["error"] = "No execution plan available"
# MAGIC         return state
# MAGIC     
# MAGIC     current_step_idx = state.get("current_step", 0)
# MAGIC     
# MAGIC     if current_step_idx >= len(plan.steps):
# MAGIC         # All steps completed
# MAGIC         return state
# MAGIC     
# MAGIC     step = plan.steps[current_step_idx]
# MAGIC     logger.info(f"Executing step {step.step_number}: {step.action} - {step.reasoning}")
# MAGIC     
# MAGIC     try:
# MAGIC         result = None
# MAGIC         
# MAGIC         if step.action == "tool_call":
# MAGIC             # Find and execute the tool
# MAGIC             tool = next((t for t in tools if t.name == step.tool_name), None)
# MAGIC             if not tool:
# MAGIC                 available_tools = [t.name for t in tools]
# MAGIC                 error_msg = f"Tool '{step.tool_name}' not found. Available tools: {', '.join(available_tools)}"
# MAGIC                 logger.error(error_msg)
# MAGIC                 raise ValueError(error_msg)
# MAGIC             
# MAGIC             logger.info(f"Invoking tool '{step.tool_name}' with query: {step.query[:100]}...")
# MAGIC             result = tool.invoke(step.query)
# MAGIC             logger.info(f"Tool '{step.tool_name}' returned result of length: {len(str(result))}")
# MAGIC             
# MAGIC         elif step.action == "genie_query":
# MAGIC             # Execute Genie agent
# MAGIC             logger.info(f"Invoking Genie agent with query: {step.query[:100]}...")
# MAGIC             genie_result = genie_agent.invoke({"messages": [HumanMessage(content=step.query)]})
# MAGIC             
# MAGIC             # Extract the response from genie
# MAGIC             if isinstance(genie_result, dict) and "messages" in genie_result:
# MAGIC                 result = genie_result["messages"][-1].content
# MAGIC             else:
# MAGIC                 result = str(genie_result)
# MAGIC             logger.info(f"Genie agent returned result of length: {len(str(result))}")
# MAGIC                 
# MAGIC         elif step.action == "synthesize":
# MAGIC             # Synthesize results from previous steps
# MAGIC             result = "synthesis_placeholder"  # Will be handled in synthesize_response
# MAGIC             logger.info("Marked step for synthesis")
# MAGIC         
# MAGIC         # Store result
# MAGIC         if "execution_results" not in state:
# MAGIC             state["execution_results"] = {}
# MAGIC         state["execution_results"][step.step_number] = result
# MAGIC         state["current_step"] = current_step_idx + 1
# MAGIC         
# MAGIC         logger.info(f"Step {step.step_number} completed successfully")
# MAGIC         
# MAGIC     except Exception as e:
# MAGIC         logger.error(f"Step {step.step_number} failed: {str(e)}", exc_info=True)
# MAGIC         state["error"] = f"Step {step.step_number} failed: {str(e)}"
# MAGIC     
# MAGIC     return state
# MAGIC
# MAGIC @normalize_state_inputs
# MAGIC def synthesize_response(state: AgentState) -> AgentState:
# MAGIC     """Synthesize final response from execution results."""
# MAGIC     plan = state["plan"]
# MAGIC     execution_results = state["execution_results"]
# MAGIC     current_query = state["context_messages"][-1].content
# MAGIC     
# MAGIC     # Build synthesis context
# MAGIC     results_summary = []
# MAGIC     for step in plan.steps:
# MAGIC         if step.step_number in execution_results:
# MAGIC             result = execution_results[step.step_number]
# MAGIC             # Truncate long results but keep meaningful information
# MAGIC             result_str = str(result)
# MAGIC             if len(result_str) > 1000:
# MAGIC                 result_str = result_str[:1000] + "... (truncated)"
# MAGIC             results_summary.append(
# MAGIC                 f"Step {step.step_number} ({step.action}): {step.reasoning}\n"
# MAGIC                 f"Result: {result_str}"
# MAGIC             )
# MAGIC     
# MAGIC     # Check if synthesis prompt exists in config, otherwise use default
# MAGIC     synthesis_template = config['prompts']['synthesis_prompt']
# MAGIC     synthesis_prompt = synthesis_template.format(
# MAGIC         query=current_query,
# MAGIC         results="\n\n".join(results_summary),
# MAGIC         strategy=plan.overall_strategy
# MAGIC     )
# MAGIC
# MAGIC     
# MAGIC     response = llm.invoke([HumanMessage(content=synthesis_prompt)])
# MAGIC     state["final_response"] = response.content
# MAGIC     
# MAGIC     # Add the response as an AI message
# MAGIC     state["messages"].append(AIMessage(content=response.content))
# MAGIC     
# MAGIC     logger.info("Response synthesized successfully")
# MAGIC     
# MAGIC     return state
# MAGIC
# MAGIC @normalize_state_inputs
# MAGIC def should_continue_execution(state: AgentState) -> str:
# MAGIC     """Determine next step in the workflow."""
# MAGIC
# MAGIC     if not state.get("plan"):
# MAGIC         logger.error("No plan exists, cannot continue execution")
# MAGIC         return "end"
# MAGIC     # Check for errors
# MAGIC     if state.get("error"):
# MAGIC         # Only replan if we have a valid plan and haven't exceeded retries
# MAGIC         if state.get("plan") and state.get("replan_count", 0) < 2:
# MAGIC             return "replan"
# MAGIC         else:
# MAGIC             return "end"
# MAGIC     
# MAGIC     # Check if all steps are executed
# MAGIC     plan = state.get("plan")
# MAGIC     current_step = state.get("current_step", 0)
# MAGIC     
# MAGIC     if plan and current_step < len(plan.steps):
# MAGIC         return "execute"
# MAGIC     else:
# MAGIC         return "synthesize"
# MAGIC
# MAGIC @normalize_state_inputs
# MAGIC def should_replan_or_end(state: AgentState) -> str:
# MAGIC     """Determine if we should replan or end after error."""
# MAGIC     if state.get("replan_count", 0) < 2:
# MAGIC         return "replan"
# MAGIC     return "end"
# MAGIC
# MAGIC
# MAGIC ###################################################
# MAGIC ## Build LangGraph Workflow
# MAGIC ###################################################
# MAGIC
# MAGIC def create_planner_graph(checkpointer: PostgresSaver) -> StateGraph:
# MAGIC     """Create the planner-based workflow graph."""
# MAGIC     
# MAGIC     workflow = StateGraph(AgentState)
# MAGIC     
# MAGIC     # Add nodes
# MAGIC     workflow.add_node("classify", classify_topic)
# MAGIC     workflow.add_node("plan", create_plan)
# MAGIC     workflow.add_node("execute", execute_step)
# MAGIC     workflow.add_node("replan", replan)
# MAGIC     workflow.add_node("synthesize", synthesize_response)
# MAGIC     
# MAGIC     # Define edges
# MAGIC     workflow.set_entry_point("classify")
# MAGIC     workflow.add_edge("classify", "plan")
# MAGIC     workflow.add_edge("plan", "execute")
# MAGIC     
# MAGIC     # Conditional routing after execution
# MAGIC     workflow.add_conditional_edges(
# MAGIC         "execute",
# MAGIC         should_continue_execution,
# MAGIC         {
# MAGIC             "execute": "execute",  # Loop back for next step
# MAGIC             "synthesize": "synthesize",
# MAGIC             "replan": "replan",
# MAGIC             "end": END
# MAGIC         }
# MAGIC     )
# MAGIC     
# MAGIC     workflow.add_edge("replan", "execute")
# MAGIC     workflow.add_edge("synthesize", END)
# MAGIC     
# MAGIC     return workflow.compile(checkpointer=checkpointer)
# MAGIC
# MAGIC
# MAGIC ###################################################
# MAGIC ## Main ResponsesAgent Implementation
# MAGIC ###################################################
# MAGIC
# MAGIC class PlannerResponsesAgent(ResponsesAgent):
# MAGIC     """Planner-based multi-agent using ResponsesAgent with PostgreSQL checkpointing."""
# MAGIC     
# MAGIC     def __init__(self, lakebase_config: dict[str, Any]):
# MAGIC         self.lakebase_config = lakebase_config
# MAGIC         self.workspace_client = WorkspaceClient()
# MAGIC         
# MAGIC         # Initialize state manager
# MAGIC         self.state_manager = DatabricksStateManager(lakebase_config=lakebase_config)
# MAGIC         
# MAGIC         mlflow.langchain.autolog()
# MAGIC     
# MAGIC     def _langchain_to_responses(self, message: BaseMessage) -> list[dict[str, Any]]:
# MAGIC         """Convert LangChain message to Responses output items."""
# MAGIC         message_dict = message.model_dump()
# MAGIC         role = message_dict["type"]
# MAGIC         output = []
# MAGIC         
# MAGIC         if role == "ai":
# MAGIC             # Handle text content
# MAGIC             content = message_dict.get("content", "")
# MAGIC             if content and isinstance(content, str):
# MAGIC                 output.append(
# MAGIC                     self.create_text_output_item(
# MAGIC                         text=content,
# MAGIC                         id=message_dict.get("id") or str(uuid.uuid4()),
# MAGIC                     )
# MAGIC                 )
# MAGIC             
# MAGIC             # Handle tool calls
# MAGIC             if tool_calls := message_dict.get("tool_calls"):
# MAGIC                 output.extend([
# MAGIC                     self.create_function_call_item(
# MAGIC                         id=message_dict.get("id") or str(uuid.uuid4()),
# MAGIC                         call_id=tool_call["id"],
# MAGIC                         name=tool_call["name"],
# MAGIC                         arguments=json.dumps(tool_call["args"])
# MAGIC                     )
# MAGIC                     for tool_call in tool_calls
# MAGIC                 ])
# MAGIC         
# MAGIC         elif role == "tool":
# MAGIC             output.append(
# MAGIC                 self.create_function_call_output_item(
# MAGIC                     call_id=message_dict["tool_call_id"],
# MAGIC                     output=message_dict["content"],
# MAGIC                 )
# MAGIC             )
# MAGIC         
# MAGIC         return output
# MAGIC     
# MAGIC     def _create_planning_status_message(self, plan: Plan) -> dict[str, Any]:
# MAGIC         """Create a user-friendly status message showing the plan."""
# MAGIC         
# MAGIC         # Create a conversational, high-level summary
# MAGIC         action_verbs = {
# MAGIC             "tool_call": "searching",
# MAGIC             "genie_query": "analyzing",
# MAGIC             "synthesize": "compiling"
# MAGIC         }
# MAGIC         
# MAGIC         # Count different action types
# MAGIC         search_steps = sum(1 for s in plan.steps if s.action == "tool_call")
# MAGIC         genie_steps = sum(1 for s in plan.steps if s.action == "genie_query")
# MAGIC         
# MAGIC         # Build friendly message
# MAGIC         plan_text = "💡 **I understand your question! Let me help you with that.**\n\n"
# MAGIC         plan_text += "I'm going to gather the information you need by checking a few different sources. "
# MAGIC         plan_text += "This should only take a moment...\n\n"
# MAGIC         plan_text += "🔄 **Starting now...**"
# MAGIC         
# MAGIC         if search_steps > 0:
# MAGIC             plan_text += f"I'll search through {search_steps} relevant data source{'s' if search_steps > 1 else ''} "
# MAGIC         if genie_steps > 0:
# MAGIC             plan_text += f"and analyze quantitative data "
# MAGIC         plan_text += "to give you a comprehensive answer.\n\n"
# MAGIC         
# MAGIC         plan_text += "🔄 **Working on it now...**"
# MAGIC         
# MAGIC         return self.create_text_output_item(
# MAGIC             text=plan_text,
# MAGIC             id=str(uuid.uuid4())
# MAGIC         )
# MAGIC     
# MAGIC     def _create_step_status_message(self, step: PlanStep, status: str) -> dict[str, Any]:
# MAGIC         """Create a user-friendly status message for step execution."""
# MAGIC         
# MAGIC         # Map actions to user-friendly messages
# MAGIC         action_messages = {
# MAGIC             "tool_call": {
# MAGIC                 "identity_search": "🔍 Checking ownership and contact information...",
# MAGIC                 "business_search": "📊 Reviewing business context and documentation...",
# MAGIC                 "technical_search": "🔧 Looking up technical specifications...",
# MAGIC                 "governance_search": "🛡️ Verifying governance and compliance details..."
# MAGIC             },
# MAGIC             "genie_query": "📈 Running data analysis...",
# MAGIC             "synthesize": "✨ Putting it all together..."
# MAGIC         }
# MAGIC         
# MAGIC         # Get appropriate message based on action type
# MAGIC         if step.action == "tool_call" and step.tool_name:
# MAGIC             status_text = action_messages["tool_call"].get(
# MAGIC                 step.tool_name,
# MAGIC                 f"🔍 Searching {step.tool_name.replace('_', ' ')}..."
# MAGIC             )
# MAGIC         elif step.action == "genie_query":
# MAGIC             status_text = action_messages["genie_query"]
# MAGIC         elif step.action == "synthesize":
# MAGIC             status_text = action_messages["synthesize"]
# MAGIC         else:
# MAGIC             status_text = f"⏳ Step {step.step_number}: {status}"
# MAGIC         
# MAGIC         return self.create_text_output_item(
# MAGIC             text=status_text,
# MAGIC             id=str(uuid.uuid4())
# MAGIC         )
# MAGIC     
# MAGIC     def predict(self, request: Union[ResponsesAgentRequest, Dict[str, Any]]) -> ResponsesAgentResponse:
# MAGIC         """Non-streaming prediction."""
# MAGIC         # Handle dict input
# MAGIC         if isinstance(request, dict):
# MAGIC             from mlflow.types.responses import ResponsesAgentRequestItem
# MAGIC             input_items = []
# MAGIC             for item in request.get("input", []):
# MAGIC                 if isinstance(item, dict):
# MAGIC                     input_items.append(ResponsesAgentRequestItem(**item))
# MAGIC                 else:
# MAGIC                     input_items.append(item)
# MAGIC             
# MAGIC             request = ResponsesAgentRequest(
# MAGIC                 input=input_items,
# MAGIC                 custom_inputs=request.get("custom_inputs", {})
# MAGIC             )
# MAGIC         
# MAGIC         ci = dict(request.custom_inputs or {})
# MAGIC         if "thread_id" not in ci:
# MAGIC             ci["thread_id"] = str(uuid.uuid4())
# MAGIC         request.custom_inputs = ci
# MAGIC         
# MAGIC         outputs = [
# MAGIC             event.item
# MAGIC             for event in self.predict_stream(request)
# MAGIC             if event.type == "response.output_item.done"
# MAGIC         ]
# MAGIC         
# MAGIC         # Include thread_id in custom outputs
# MAGIC         custom_outputs = {"thread_id": ci["thread_id"]}
# MAGIC         
# MAGIC         return ResponsesAgentResponse(output=outputs, custom_outputs=custom_outputs)
# MAGIC     
# MAGIC     def predict_stream(
# MAGIC         self,
# MAGIC         request: ResponsesAgentRequest,
# MAGIC     ) -> Generator[ResponsesAgentStreamEvent, None, None]:
# MAGIC         """Streaming prediction with plan visibility."""
# MAGIC         thread_id = (request.custom_inputs or {}).get("thread_id", str(uuid.uuid4()))
# MAGIC         # Track what we've already yielded
# MAGIC         plan_shown = False
# MAGIC         last_step_shown = -1
# MAGIC         classification_shown = False 
# MAGIC         # Convert input to LangChain messages
# MAGIC         input_messages = []
# MAGIC         for item in request.input:
# MAGIC             item_dict = item.model_dump() if hasattr(item, 'model_dump') else item
# MAGIC             role = item_dict.get("role", "user")
# MAGIC             content = item_dict.get("content", "")
# MAGIC             
# MAGIC             if role == "user":
# MAGIC                 input_messages.append(HumanMessage(content=content))
# MAGIC             elif role == "assistant":
# MAGIC                 input_messages.append(AIMessage(content=content))
# MAGIC             elif role == "system":
# MAGIC                 input_messages.append(SystemMessage(content=content))
# MAGIC         
# MAGIC         checkpoint_config = {
# MAGIC             "configurable": {"thread_id": thread_id},
# MAGIC             "recursion_limit": 50
# MAGIC         }
# MAGIC         
# MAGIC         with self.state_manager.get_connection() as conn:
# MAGIC             checkpointer = self.state_manager.create_checkpointer(conn)
# MAGIC             graph = create_planner_graph(checkpointer)
# MAGIC             
# MAGIC             # Track what we've already yielded
# MAGIC             plan_shown = False
# MAGIC             last_step_shown = -1
# MAGIC             
# MAGIC             try:
# MAGIC                 for event in graph.stream(
# MAGIC                     {"messages": input_messages},
# MAGIC                     checkpoint_config,
# MAGIC                     stream_mode="values"
# MAGIC                 ):
# MAGIC                     event = StateNormalizer.normalize_state(event)
# MAGIC                     # Show classification
# MAGIC                     if "classification" in event and event["classification"] and not classification_shown:
# MAGIC                         classification = event["classification"]
# MAGIC                         
# MAGIC                         # Create friendly classification message
# MAGIC                         if classification.is_new_topic:
# MAGIC                             message = "👋 Starting fresh with your question..."
# MAGIC                         else:
# MAGIC                             message = "🔗 Building on our previous conversation..."
# MAGIC                         
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             type="response.output_item.done",
# MAGIC                             item=self.create_text_output_item(
# MAGIC                                 text=message,
# MAGIC                                 id=str(uuid.uuid4())
# MAGIC                             )
# MAGIC                         )
# MAGIC                         classification_shown = True
# MAGIC                     
# MAGIC                     # Show plan
# MAGIC                     if "plan" in event and event["plan"] and not plan_shown:
# MAGIC                         plan = event["plan"]
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             type="response.output_item.done",
# MAGIC                             item=self._create_planning_status_message(plan)
# MAGIC                         )
# MAGIC                         plan_shown = True
# MAGIC                     
# MAGIC                     # Show step execution progress (only when step actually changes)
# MAGIC                     if "current_step" in event and "plan" in event:
# MAGIC                         current_step_idx = event["current_step"]
# MAGIC                         plan = event["plan"]
# MAGIC                         
# MAGIC                         # Only show if this is a NEW step (not a duplicate event)
# MAGIC                         if current_step_idx > last_step_shown and current_step_idx <= len(plan.steps) and current_step_idx > 0:
# MAGIC                             step = plan.steps[current_step_idx - 1]
# MAGIC                             yield ResponsesAgentStreamEvent(
# MAGIC                                 type="response.output_item.done",
# MAGIC                                 item=self._create_step_status_message(step, "Executing...")
# MAGIC                             )
# MAGIC                             last_step_shown = current_step_idx
# MAGIC                     
# MAGIC                     # Show replanning with friendly message
# MAGIC                     if "replan_count" in event and event.get("replan_count", 0) > 0:
# MAGIC                         replan_count = event["replan_count"]
# MAGIC                         
# MAGIC                         friendly_messages = {
# MAGIC                             1: "🔄 Let me try a different approach...",
# MAGIC                             2: "🔄 One more attempt with an alternative strategy..."
# MAGIC                         }
# MAGIC                         
# MAGIC                         message = friendly_messages.get(replan_count, f"🔄 Adjusting my approach (attempt {replan_count})...")
# MAGIC                         
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             type="response.output_item.done",
# MAGIC                             item=self.create_text_output_item(
# MAGIC                                 text=message,
# MAGIC                                 id=str(uuid.uuid4())
# MAGIC                             )
# MAGIC                         )
# MAGIC                         plan_shown = False  # Reset to show new plan
# MAGIC                         classification_shown = False  # Reset classification
# MAGIC                     
# MAGIC                     # Return final response (only the latest AI message)
# MAGIC                     if "final_response" in event and event["final_response"]:
# MAGIC                         # Only return the latest AI message
# MAGIC                         messages = event.get("messages", [])
# MAGIC                         if messages and isinstance(messages[-1], AIMessage):
# MAGIC                             items = self._langchain_to_responses(messages[-1])
# MAGIC                             for item in items:
# MAGIC                                 yield ResponsesAgentStreamEvent(
# MAGIC                                     type="response.output_item.done",
# MAGIC                                     item=item
# MAGIC                                 )
# MAGIC                     
# MAGIC                     # Handle errors
# MAGIC                     if "error" in event and event["error"]:
# MAGIC                         yield ResponsesAgentStreamEvent(
# MAGIC                             type="response.output_item.done",
# MAGIC                             item=self.create_text_output_item(
# MAGIC                                 text=f"❌ Error: {event['error']}",
# MAGIC                                 id=str(uuid.uuid4())
# MAGIC                             )
# MAGIC                         )
# MAGIC                         
# MAGIC             except Exception as e:
# MAGIC                 logger.error(f"Error during agent streaming: {e}", exc_info=True)
# MAGIC                 yield ResponsesAgentStreamEvent(
# MAGIC                     type="response.output_item.done",
# MAGIC                     item=self.create_text_output_item(
# MAGIC                         text=f"❌ Fatal Error: {str(e)}",
# MAGIC                         id=str(uuid.uuid4())
# MAGIC                     )
# MAGIC                 )
# MAGIC                 raise
# MAGIC
# MAGIC
# MAGIC # ----- Export model -----
# MAGIC AGENT = PlannerResponsesAgent(config["lakebase_config"])
# MAGIC mlflow.models.set_model(AGENT)
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Test the agent
# MAGIC
# MAGIC Interact with the agent to test its output. Since this notebook called `mlflow.langchain.autolog()` you can view the trace for each step the agent takes.
# MAGIC
# MAGIC Replace this placeholder input with an appropriate domain-specific example for your agent.

# COMMAND ----------

# DBTITLE 1,Restart python
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Validate the code
from agentv1_7 import AGENT


 
# # Message 1, don't include thread_id (creates new thread)
result = AGENT.predict({
    "input": [{"role": "user", "content": "Who owns the wells_master table and how can I contact them?"}]
})
print(result.model_dump(exclude_none=True))
thread_id = result.custom_outputs["thread_id"]

# thread_id = ''
response2 = AGENT.predict({
    "input": [{"role": "user", "content": "what is the business purpose of the underlying schema"}],
    "custom_inputs": {"thread_id": thread_id}
})
print("Response 2:", response2.model_dump(exclude_none=True))



response3 = AGENT.predict({
    "input": [{"role": "user", "content": "total cost by workspace id"}],
    "custom_inputs": {"thread_id": thread_id}
})

print("Response 3:", response3.model_dump(exclude_none=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Log the `agent` as an MLflow model
# MAGIC Determine Databricks resources to specify for automatic auth passthrough at deployment time
# MAGIC - **TODO**: If your Unity Catalog Function queries a [vector search index](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/unstructured-retrieval-tools) or leverages [external functions](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/external-connection-tools), you need to include the dependent vector search index and UC connection objects, respectively, as resources. See [docs](https://learn.microsoft.com/azure/databricks/generative-ai/agent-framework/log-agent#specify-resources-for-automatic-authentication-passthrough) for more details.
# MAGIC
# MAGIC Log the agent as code from the `agent.py` file. See [MLflow - Models from Code](https://mlflow.org/docs/latest/models.html#models-from-code).

# COMMAND ----------

import yaml
config ={}
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Filter out objects that not accessible from the current workspace
# MAGIC * Scan through the metric view to extract source text.
# MAGIC * Extract catalogs from the sql text
# MAGIC * Find a list of inaccessible objects

# COMMAND ----------

# DBTITLE 1,Get list of all table objects to be added as Resources
import sqlparse
import re
from sqlparse.sql import IdentifierList, Identifier
from sqlparse.tokens import Keyword, Name

def extract_catalog_names(sql_statement):
    """
    Extract catalog names from Databricks SQL statements.
    
    Args:
        sql_statement (str): The SQL statement to parse
        
    Returns:
        set: Set of unique catalog names found in the SQL statement
    """
    catalogs = set()
    
    # Normalize the SQL to handle complex formatting
    sql_normalized = re.sub(r'\s+', ' ', sql_statement.strip())
    
    # Find all three-part identifiers (catalog.schema.table)
    # This pattern looks for word.word.word that are likely table references
    three_part_pattern = r'\b([a-zA-Z_`"\[][\w`"\[\]]*)\s*\.\s*([a-zA-Z_`"\[][\w`"\[\]]*)\s*\.\s*([a-zA-Z_`"\[][\w`"\[\]]*)\b'
    three_part_matches = re.findall(three_part_pattern, sql_normalized, re.IGNORECASE)
    
    # Extract potential catalogs from three-part identifiers
    potential_catalogs = set()
    for match in three_part_matches:
        catalog = match[0].strip('`"[]')
        if catalog:
            potential_catalogs.add(catalog)
    
    # Now we need to filter out aliases by looking at the context
    # Step 1: Find all table aliases in FROM and JOIN clauses
    aliases = set()
    
    # Pattern to match table aliases in FROM/JOIN clauses
    # This looks for: FROM/JOIN table_reference alias_name
    alias_patterns = [
        r'\bFROM\s+(?:[a-zA-Z_`"\[][\w`"\[\].]*\s+)(\w+)(?:\s+(?:LEFT|RIGHT|INNER|OUTER|JOIN|WHERE|GROUP|ORDER|HAVING|,)|\s*$)',
        r'\b(?:LEFT|RIGHT|INNER|OUTER)?\s*JOIN\s+(?:[a-zA-Z_`"\[][\w`"\[\].]*\s+)(\w+)(?:\s+(?:ON|LEFT|RIGHT|INNER|OUTER|JOIN|WHERE|GROUP|ORDER|HAVING|,)|\s*$)'
    ]
    
    for pattern in alias_patterns:
        matches = re.findall(pattern, sql_normalized, re.IGNORECASE)
        for match in matches:
            # Only consider short names as aliases (typically 1-4 characters)
            if len(match) <= 4 and match.isalpha():
                aliases.add(match.lower())
    
    # Step 2: Look for dot notation that suggests aliases (alias.column)
    # Pattern for two-part identifiers that are likely alias.column
    two_part_pattern = r'\b([a-zA-Z_][\w]*)\s*\.\s*([a-zA-Z_][\w]*)\s*(?!\s*\.)'
    two_part_matches = re.findall(two_part_pattern, sql_normalized, re.IGNORECASE)
    
    for match in two_part_matches:
        potential_alias = match[0]
        # If it's short and used in a column context, likely an alias
        if len(potential_alias) <= 4:
            aliases.add(potential_alias.lower())
    
    # Step 3: Filter out aliases from potential catalogs
    for catalog in potential_catalogs:
        if catalog.lower() not in aliases:
            # Additional validation: catalog names are usually meaningful words
            # and not single letters or very short abbreviations in this context
            if len(catalog) >= 3 or catalog.lower() in ['system']:  # 'system' is a known Databricks catalog
                catalogs.add(catalog)
    
    # Step 4: Handle special cases - look for quoted three-part identifiers
    quoted_pattern = r'`([^`]+)`\.`([^`]+)`\.`([^`]+)`'
    quoted_matches = re.findall(quoted_pattern, sql_normalized)
    for match in quoted_matches:
        catalog = match[0]
        if catalog and catalog.lower() not in aliases:
            catalogs.add(catalog)
    
    # Step 5: Additional cleanup - remove common SQL keywords that might be misidentified
    sql_keywords = {'select', 'from', 'where', 'join', 'on', 'and', 'or', 'as', 'with', 'union', 'all'}
    catalogs = [cat for cat in catalogs if cat.lower() not in sql_keywords]
    
    return catalogs

import yaml
import sqlparse
from databricks.sdk import WorkspaceClient
client = WorkspaceClient()
catalogs = client.catalogs.list()
catalog_list = []
for catalog in catalogs:
    catalog_list.append(catalog.name)

inaccessible_catalogs = []
inaccessible_objects = []
table_list = client.tables.list(catalog_name= config['catalog'], schema_name= config['tools']['genie_table_schema'])
for table in table_list: 
    _inaccessible_catalogs = []
    if str(table.table_type) == "TableType.METRIC_VIEW":
        view_config = yaml.safe_load(table.view_definition)
        catalogs = extract_catalog_names(view_config['source'])
        _inaccessible_catalogs = [x for x in catalogs if x not in catalog_list]
        inaccessible_catalogs.extend([x for x in _inaccessible_catalogs if x not in inaccessible_catalogs])
        if len(_inaccessible_catalogs) > 0:
            inaccessible_objects.append(table.catalog_name+'.'+table.schema_name+'.'+table.name)

print(f"The following catalogs are not accessible in the workspace: {inaccessible_catalogs}")
str_objects = 'The following objects will be excluded from the ML Flow Resource registration:'
for table in inaccessible_objects:
  str_objects += f"\n * {table}"
print(f"Excluding these objects from ML Flow Resource registration: {str_objects}")

# COMMAND ----------

# DBTITLE 1,Validate Manage Access on Tables
_user = spark.sql("select current_user()").collect()[0][0]
for table_name in table_list:
    _df = spark.sql(f"show grants on table {table_name}")
    if _df.filter("ActionType = 'MANAGE'").filter(f"Principal = '{_user}'").count() < 1:
        print(f"No manage access on table {table_name}")
    # print(f"table_name: {table_name} : {_df.filter("ActionType = 'MANAGE'").filter(f"Principal = '{_user}'").count()}")

# COMMAND ----------

# DBTITLE 1,Add Resources and Log Model
# Determine Databricks resources to specify for automatic auth passthrough at deployment time
import mlflow
from agentv1_7 import LLM_ENDPOINT_NAME, tools
from databricks_langchain import VectorSearchRetrieverTool

from pkg_resources import get_distribution
from unitycatalog.ai.langchain.toolkit import UnityCatalogTool
from mlflow.models.resources import (
  DatabricksVectorSearchIndex,
  DatabricksServingEndpoint,
  DatabricksSQLWarehouse,
  DatabricksFunction,
  DatabricksGenieSpace,
  DatabricksTable,
  DatabricksUCConnection,
  DatabricksApp,
  DatabricksLakebase
)
from databricks.sdk import WorkspaceClient
client = WorkspaceClient()



table_resources = []
table_list = []


_table_list = client.tables.list(catalog_name= config['catalog'], schema_name= config['tools']['genie_table_schema'])
for table in _table_list:
    table_list.append(f"{table.catalog_name}.{table.schema_name}.{table.name}")

print(f"Total table list : {len(table_list)}")
table_list = [x for x in table_list if x not in inaccessible_objects]

print(f"Table list after removing inaccessible tables : {len(table_list)}")
for table in table_list:
    table_resources.append(DatabricksTable(table_name=table))


resources = [
    DatabricksServingEndpoint(endpoint_name=config['llm_endpoint']),
    DatabricksServingEndpoint(endpoint_name=config['tools']['vector_search_embedding_endpoint']),
    DatabricksVectorSearchIndex(index_name=config['tools']['vector_search_identity_index']),
    DatabricksVectorSearchIndex(index_name=config['tools']['vector_search_technical_index']),
    DatabricksVectorSearchIndex(index_name=config['tools']['vector_search_business_index']),
    DatabricksVectorSearchIndex(index_name=config['tools']['vector_search_governance_index']),
    DatabricksGenieSpace(genie_space_id=config['tools']['genie_space_id']),
    DatabricksLakebase(database_instance_name=config['lakebase_config']["instance_name"])
]
resources.extend(table_resources)

for tool in tools:
    if isinstance(tool, VectorSearchRetrieverTool):
        resources.extend(tool.resources)
    elif isinstance(tool, UnityCatalogTool):
        # TODO: If the UC function includes dependencies like external connection or vector search, please include them manually.
        # See the TODO in the markdown above for more information.
        resources.append(DatabricksFunction(function_name=tool.uc_function_name))

input_example = {
    "input": [
        {
            "role": "user",
            "content": "Who owns the wells_master table and how can I contact them?"
        }
    ]
}

with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="agentv1_7",
        python_model="agentv1_7.py",
        model_config='config.yaml',
        input_example=input_example,
        code_paths=[
             "lb_state_manager.py","message_converters.py"
        ],
        resources=resources,
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
            f"psycopg[binary,pool]",
            f"langgraph_supervisor",
            f"pydantic=={get_distribution('pydantic').version}",
            f"langgraph-checkpoint-postgres=={get_distribution('langgraph-checkpoint-postgres').version}",
        ],
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evaluate the agent with [Agent Evaluation](https://learn.microsoft.com/azure/databricks/mlflow3/genai/eval-monitor/)
# MAGIC
# MAGIC You can edit the requests or expected responses in your evaluation dataset and run evaluation as you iterate your agent, leveraging mlflow to track the computed quality metrics.
# MAGIC
# MAGIC Evaluate your agent with one of our [predefined LLM scorers](https://learn.microsoft.com/azure/databricks/mlflow3/genai/eval-monitor/predefined-judge-scorers), or try adding [custom metrics](https://learn.microsoft.com/azure/databricks/mlflow3/genai/eval-monitor/custom-scorers).

# COMMAND ----------

# DBTITLE 1,Basic Evaluation
import mlflow
from agentv1_7 import AGENT
from mlflow.genai.scorers import RelevanceToQuery, Safety, RetrievalRelevance, RetrievalGroundedness

# Updated dataset - the 'inputs' dictionary keys must match predict_fn parameter names
eval_dataset = [
    {
        "inputs": {
            "input": [
                {
                    "role": "user",
                    "content": "Who owns the wells_master table and how can I contact them?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "input": [
                {
                    "role": "user", 
                    "content": "What is the business purpose of the reservoir_management schema and which projects use it?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "input": [
                {
                    "role": "user",
                    "content": "Show me all tables classified as 'Highly Confidential' and what are their retention policies?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "input": [
                {
                    "role": "user",
                    "content": "What source systems feed data into the daily_production table and how often is it refreshed?"
                }
            ]
        },
        "expected_response": None
    },
    {
        "inputs": {
            "input": [
                {
                    "role": "user",
                    "content": "I need to find all high-criticality financial data owned by the Finance team that comes from SAP systems. Who should I contact about access, and what compliance requirements apply?"
                }
            ]
        },
        "expected_response": None
    }
]

# Predict function parameter name must match the key in 'inputs' dictionary
def predict_fn(input):
    """Convert input format and call the ResponsesAgent"""
    return AGENT.predict({"input": input})

eval_results = mlflow.genai.evaluate(
    data=eval_dataset,
    predict_fn=predict_fn,
    scorers=[RelevanceToQuery(), Safety()],
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Perform pre-deployment validation of the agent
# MAGIC Before registering and deploying the agent, we perform pre-deployment checks via the [mlflow.models.predict()](https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.predict) API. See [documentation](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/model-serving-debug#validate-inputs) for details

# COMMAND ----------

# DBTITLE 1,Pre Deployment Evaluation
mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/agentv1_7",
    input_data={"input": [{"role": "user", "content": "Show me cost by workspacenames"}]},
    env_manager="uv",
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register the model to Unity Catalog
# MAGIC
# MAGIC Update the `catalog`, `schema`, and `model_name` below to register the MLflow model to Unity Catalog.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = ""
schema = ""
model_name = ""
UC_MODEL_NAME = f"yourcatalog.yourschema.supervisor_v_1_7"


# register the model to UC
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME
)

# COMMAND ----------

# DBTITLE 1,Verify the Run Number and Latest Version
import mlflow
from mlflow.tracking import MlflowClient

client = MlflowClient() 

# Get all versions of the model
all_versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")

if all_versions:
    # Sort by version number to get the latest
    latest_version = max(all_versions, key=lambda x: int(x.version))
    
    version_number = latest_version.version
    run_id = latest_version.run_id
    
    print(f"Latest Version Number: {version_number}")
    print(f"Run ID: {run_id}")
else:
    print(f"No versions found for model: {UC_MODEL_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy the agent

# COMMAND ----------

# DBTITLE 1,Deploy the Agent
from databricks import agents
deployment_info = agents.deploy(UC_MODEL_NAME, uc_registered_model_info.version, tags = {"endpointSource": "playground"}, endpoint_name="yourendpointname")

# COMMAND ----------

# DBTITLE 1,Check if the deployment completed
from databricks.sdk import WorkspaceClient
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
client = WorkspaceClient()

while client.serving_endpoints.get(deployment_info.endpoint_name).state.ready == EndpointStateReady.NOT_READY or client.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
    if client.serving_endpoints.get(deployment_info.endpoint_name).state.config_update == EndpointStateConfigUpdate.UPDATE_FAILED:
        break
    print(client.serving_endpoints.get(deployment_info.endpoint_name).state.as_dict())
    time.sleep(30)


# COMMAND ----------

# DBTITLE 1,Warm Up endpoint if scaled to 0
from databricks.sdk import WorkspaceClient
import time
from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
import requests
import os

def warmup_endpoint(endpoint_name, token, workspace_url, interval=30):
    """
    Warm up a Databricks serving endpoint if it is scaled to zero.
    
    Args:
        endpoint_name (str): Name of the serving endpoint.
        token (str): Databricks workspace access token.
        workspace_url (str): Databricks workspace URL.
        interval (int): Interval in seconds between warm-up requests.
    """
    client = WorkspaceClient()
    endpoint = client.serving_endpoints.get(endpoint_name)
    print("Served Entities:")
    for entity in endpoint.config.served_entities:
        print(f"- Entity Name: {entity.entity_name}, Workload Size: {entity.workload_size}, Scale to Zero: {entity.scale_to_zero_enabled}")

    # Get the latest served entity
    latest_entity = None
    if hasattr(endpoint, "config") and hasattr(endpoint.config, "served_entities") and endpoint.config.served_entities:
        if hasattr(endpoint.config.served_entities[0], "creation_timestamp"):
            latest_entity = max(endpoint.config.served_entities, key=lambda e: getattr(e, "creation_timestamp", 0))
        elif hasattr(endpoint.config.served_entities[0], "last_updated_timestamp"):
            latest_entity = max(endpoint.config.served_entities, key=lambda e: getattr(e, "last_updated_timestamp", 0))
        else:
            latest_entity = endpoint.config.served_entities[-1]
        print("\nLatest Served Entity:")

    print(endpoint.state.config_update)

    # Check if latest_entity is scaled to zero using deployment_state_message
    if latest_entity and hasattr(latest_entity, "state") and hasattr(latest_entity.state, "deployment_state_message"):
        deployment_state_message = latest_entity.state.deployment_state_message
        if "scaled to zero" in deployment_state_message.lower():
            print("Latest served entity is scaled to zero. Sending warm-up messages every 30 seconds until deployment state changes...")
            endpoint_url = f"{workspace_url.rstrip('/')}/serving-endpoints/{endpoint_name}/invocations"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            warmup_payload = {
                "input": [{"role": "user", "content": "ping"}]
            }
            while True:
                # Refresh endpoint and latest_entity state
                endpoint = client.serving_endpoints.get(endpoint_name)
                if hasattr(endpoint, "config") and hasattr(endpoint.config, "served_entities") and endpoint.config.served_entities:
                    if hasattr(endpoint.config.served_entities[0], "creation_timestamp"):
                        latest_entity = max(endpoint.config.served_entities, key=lambda e: getattr(e, "creation_timestamp", 0))
                    elif hasattr(endpoint.config.served_entities[0], "last_updated_timestamp"):
                        latest_entity = max(endpoint.config.served_entities, key=lambda e: getattr(e, "last_updated_timestamp", 0))
                    else:
                        latest_entity = endpoint.config.served_entities[-1]
                if hasattr(latest_entity, "state") and hasattr(latest_entity.state, "deployment_state_message"):
                    new_message = latest_entity.state.deployment_state_message
                    print(f"Current deployment_state_message: {new_message}")
                    if "scaled to zero" not in new_message.lower():
                        print("Deployment state changed. Warm-up complete.")
                        break
                try:
                    resp = requests.post(endpoint_url, headers=headers, json=warmup_payload, timeout=60)
                    print(f"Warm-up request sent. Status: {resp.status_code}")
                except Exception as e:
                    print(f"Warm-up request failed: {e}")
                time.sleep(interval)
            print("Warm-up complete.")
        else:
            print("Latest served entity is not scaled to zero. No warm-up needed.")

# Example usage:
TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
WORKSPACE_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('browserHostName')
warmup_endpoint(endpoint_name='yourendpointname', token=TOKEN, workspace_url=WORKSPACE_URL)

# COMMAND ----------

# DBTITLE 1,Test the agent after deploying from Model Serving
import requests
import json
import os
from typing import Dict, Optional, Any, Iterator


class DatabricksAgentClient:
    """
    Client for calling Databricks Agent serving endpoints via HTTP with streaming support.
    Supports state management through thread_id for conversational context.
    Uses NDJSON (newline-delimited JSON) streaming format.
    """
    
    def __init__(self, workspace_url: str, token: str, endpoint_name: str):
        """
        Initialize the Databricks Agent client.
        
        Args:
            workspace_url: Databricks workspace URL (e.g., 'https://adb-xxx.azuredatabricks.net')
            token: Databricks workspace access token
            endpoint_name: Name of the deployed serving endpoint
        """
        self.workspace_url = workspace_url.rstrip('/')
        self.token = token
        self.endpoint_name = endpoint_name
        self.endpoint_url = f"{self.workspace_url}/serving-endpoints/{endpoint_name}/invocations"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        self.thread_id = None
    
    def predict_stream(
        self, 
        user_input: str, 
        custom_inputs: Optional[Dict[str, Any]] = None,
        role: str = "user"
    ) -> Iterator[Dict[str, Any]]:
        """
        Send a prediction request with streaming response.
        
        Args:
            user_input: The user's question or message
            custom_inputs: Optional custom inputs including thread_id for state management
            role: Role of the message sender (default: "user")
            
        Yields:
            Dictionary containing each streamed event from the agent
        """
        payload = {
            "input": [{"role": role, "content": user_input}]
        }
        
        # Add custom inputs if provided (e.g., thread_id for conversation context)
        if custom_inputs:
            payload["custom_inputs"] = custom_inputs
        
        # Databricks uses NDJSON streaming, not SSE
        stream_headers = self.headers.copy()
        
        try:
            print(f"Payload:{payload}")
            response = requests.post(
                self.endpoint_url,
                headers=stream_headers,
                json=payload,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            # Process newline-delimited JSON stream
            for line in response.iter_lines(decode_unicode=True):
                if line:  # Skip empty lines
                    try:
                        event_data = json.loads(line)
                        
                        # Capture thread_id if present in custom_outputs
                        if "custom_outputs" in event_data and "thread_id" in event_data["custom_outputs"]:
                            self.thread_id = event_data["custom_outputs"]["thread_id"]
                        
                        yield event_data
                        
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line as JSON: {line}")
                        print(f"Error: {e}")
                        yield {"raw_data": line}
            
        except requests.exceptions.RequestException as e:
            print(f"Error calling endpoint: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            raise
    
    def chat_stream(self, user_input: str, use_context: bool = True) -> Iterator[Dict[str, Any]]:
        """
        Send a chat message with streaming and automatic context management.
        
        Args:
            user_input: The user's question or message
            use_context: Whether to use the existing conversation context (default: True)
            
        Yields:
            Dictionary containing each streamed event from the agent
        """
        custom_inputs = None
        if use_context and self.thread_id:
            custom_inputs = {"thread_id": self.thread_id}
        
        yield from self.predict_stream(user_input, custom_inputs=custom_inputs)
    
    def reset_context(self):
        """Reset the conversation context by clearing the thread_id."""
        self.thread_id = None
        print("Conversation context reset.")


def main():
    """
    Example usage demonstrating streaming responses with your sample questions.
    """
    # Configuration - replace with your actual values or use environment variables
    WORKSPACE_URL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()
    TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
    ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "yourendpointname")
    
    if TOKEN == "your-token-here":
        print("Please set DATABRICKS_TOKEN environment variable or update the script")
        return
    
    # Initialize the client
    agent = DatabricksAgentClient(
        workspace_url=WORKSPACE_URL,
        token=TOKEN,
        endpoint_name=ENDPOINT_NAME
    )
    
    print("=" * 80)
    print("Databricks Agent Client - Streaming Conversational Testing")
    print("=" * 80)
    
    # Question 1: Initial question - establishes context
    print("\n[Question 1] Who owns the reservoir pressure data?")
    print("-" * 80)
    print("Streaming response:\n")
    
    for event in agent.predict_stream("Who owns the reservoir pressure data?"):
        print(json.dumps(event, indent=2, default=str))
        print("-" * 40)
    
    print(f"\n✓ Thread ID captured: {agent.thread_id}\n")
    
    # Question 2: Follow-up question - should use context from Q1
    print("\n[Question 2] What's the business purpose of the well performance metrics table")
    print("-" * 80)
    print("Streaming response:\n")
    
    for event in agent.chat_stream("What's the business purpose of the well performance metrics table"):
        print(json.dumps(event, indent=2, default=str))
        print("-" * 40)
    
    # Question 3: Context switch - different topic about costs
    print("\n[Question 3] How fresh is the facility operations data? When was it last updated?")
    print("-" * 80)
    print("Streaming response:\n")
    
    for event in agent.chat_stream("How fresh is the facility operations data? When was it last updated?"):
        print(json.dumps(event, indent=2, default=str))
        print("-" * 40)

    # Question 4: Context switch - different topic about costs
    print("\n[Question 4] Which production tables are sourced from the same system as daily_production?")
    print("-" * 80)
    print("Streaming response:\n")
    
    for event in agent.chat_stream("Which production tables are sourced from the same system as daily_production?"):
        print(json.dumps(event, indent=2, default=str))
        print("-" * 40)

    # Question 5: Governance & Compliance Questions
    print("\n[Question 5] What's the data classification and retention policy for safety incidents?")
    print("-" * 80)
    print("Streaming response:\n")
    
    for event in agent.chat_stream("What's the data classification and retention policy for safety incidents?"):
        print(json.dumps(event, indent=2, default=str))
        print("-" * 40)

    # Question 6: Context switch - different topic about costs
    print("\n[Question 6] Show me all Highly Confidential tables that contain SEC-reportable financial data?")
    print("-" * 80)
    print("Streaming response:\n")
    
    for event in agent.chat_stream("Show me all Highly Confidential tables that contain SEC-reportable financial data"):
        print(json.dumps(event, indent=2, default=str))
        print("-" * 40)

    # Question 6: Governance & Compliance Questions
    print("\n[Question 6] I need to analyze Permian Basin well performance. Find relevant tables, check data quality, and generate a starter query.")
    print("-" * 80)
    print("Streaming response:\n")
    
    for event in agent.chat_stream("I need to analyze Permian Basin well performance. Find relevant tables, check data quality, and generate a starter query."):
        print(json.dumps(event, indent=2, default=str))
        print("-" * 40)

    print("\n" + "=" * 80)
    print("All questions completed with streaming responses")
    print("=" * 80)


main()
