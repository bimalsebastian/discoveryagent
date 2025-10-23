# Intelligent Planning Agent for Data Discovery

> Conversational AI agent that plans before it acts. Built natively on Databricks with domain-split vector search, intelligent replanning, and persistent conversation state.

[![Databricks](https://img.shields.io/badge/Databricks-Native-FF3621?logo=databricks)](https://databricks.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Workflows-blue)](https://langchain-ai.github.io/langgraph/)
[![MLflow](https://img.shields.io/badge/MLflow-3.x-0194E2?logo=mlflow)](https://mlflow.org)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## 🎯 What Is This?

An AI agent that answers complex data questions by **planning multi-step strategies** instead of hoping keyword search works.

**Ask it:**
> "Who owns customer_data, what's it for, and how much did queries cost last month?"

**Get:**
> A coherent answer synthesized from multiple systems (not 50 search results to sift through)

### Traditional Search vs. This Agent

| Traditional Approach | This Agent |
|---------------------|------------|
| Keyword search → 47 documents | Plans → Executes → Synthesizes |
| Single search index | 4 specialized domain indices |
| No conversation memory | Full context across sessions |
| Manual error recovery | Automatic replanning |
| Static retrieval | Dynamic tool selection |

---

## ✨ Key Features

### 🧠 **Planning-First Architecture**
- Creates structured execution plans before taking action
- Routes queries to appropriate tools (Vector Search, Analytics, etc.)
- Transparent reasoning process visible to users

### 🎯 **Domain-Split Vector Search**
Four specialized indices instead of one monolithic search:
- **Identity & Ownership** - Who owns/manages data
- **Business Context** - Projects, domains, use cases
- **Technical Specifications** - Schemas, columns, architecture
- **Governance & Compliance** - Classifications, policies, regulations

### 🔄 **Intelligent Replanning**
- Automatically adapts when initial plans fail
- Up to 2 replanning attempts with learned context
- User-friendly error recovery

### 💬 **Persistent Conversations**
- PostgreSQL-based state checkpointing
- Resume conversations after hours, days, or weeks
- Full history preservation with thread-based isolation

### 🏗️ **100% Databricks Native**
- Databricks Vector Search
- Genie Spaces for analytics
- Lakebase PostgreSQL for state
- Unity Catalog integration
- Model Serving deployment

---

## 🚀 Quick Start

### Prerequisites

- Databricks workspace (AWS, Azure, or GCP)
- Unity Catalog enabled
- Access to:
  - Model Serving endpoint
  - Vector Search
  - Lakebase PostgreSQL instance
  - Genie Space (optional for analytics)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-org/intelligent-planning-agent.git
cd intelligent-planning-agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Configure settings**
```bash
cp config.yaml.example config.yaml
# Edit config.yaml with your Databricks settings
```

4. **Set up vector indices**

Run the data pipeline notebook to create domain-split indices:
```bash
databricks workspace import notebooks/Vector_Search_Setup.py
# Execute the notebook in your workspace
```

5. **Initialize PostgreSQL checkpointing**
```python
from lb_state_manager import DatabricksStateManager

state_manager = DatabricksStateManager(lakebase_config={
    "instance_name": "your-instance",
    "conn_host": "instance-xxx.database.azuredatabricks.net",
    "conn_db_name": "your_database",
    "workspace_host": "https://your-workspace.azuredatabricks.net"
})

state_manager.initialize_checkpoint_tables()
```

6. **Deploy to Model Serving**
```python
import mlflow

mlflow.models.log_model(
    python_model=agent,
    artifact_path="planning_agent",
    registered_model_name="planning_agent"
)
```

---

## 📖 Architecture

### High-Level Flow

```
User Query
    ↓
Classify Topic (New or Continuation)
    ↓
Create Plan (Multi-step Strategy)
    ↓
Execute Steps (Loop)
    ├─→ Vector Search (Domain-specific)
    ├─→ Genie Analytics
    └─→ Other Tools
    ↓
Synthesize Response
    ↓
Save State (PostgreSQL)
```

### Planning Example

**User Query:** "Who owns customer_analytics and what's its business purpose?"

**Agent's Plan:**
```python
Plan(
    steps=[
        PlanStep(
            action="tool_call",
            tool_name="identity_search",
            query="customer_analytics ownership contact",
            reasoning="Retrieve owner information"
        ),
        PlanStep(
            action="tool_call",
            tool_name="business_search",
            query="customer_analytics business purpose",
            reasoning="Retrieve business context"
        ),
        PlanStep(
            action="synthesize",
            reasoning="Combine results into coherent answer"
        )
    ],
    overall_strategy="Search identity and business indices for complete information"
)
```

### Domain-Split Vector Indices

Instead of one monolithic index, metadata is organized into specialized domains:

| Domain | Focus | Example Queries |
|--------|-------|----------------|
| **Identity** | Ownership, contacts, teams | "Who owns X?" |
| **Business** | Projects, use cases, context | "What's X used for?" |
| **Technical** | Schemas, columns, architecture | "How is X structured?" |
| **Governance** | Compliance, classifications | "What's the classification?" |

**Why?** Each index is optimized for specific query patterns, dramatically improving retrieval precision.

---

## 🔧 Configuration

### config.yaml Structure

```yaml
# LLM Configuration
llm_endpoint: databricks-claude-3-7-sonnet

# Vector Search Indices
tools:
  vector_search_identity_index: catalog.schema.metadata_identity_idx
  vector_search_business_index: catalog.schema.metadata_business_idx
  vector_search_technical_index: catalog.schema.metadata_technical_idx
  vector_search_governance_index: catalog.schema.metadata_governance_idx
  vector_search_embedding_endpoint: databricks-bge-large-en
  
  # Genie Configuration
  genie_space_id: your-genie-space-id
  genie_agent_name: Genie

# PostgreSQL Lakebase
lakebase_config:
  instance_name: your-instance-name
  conn_host: instance-xxx.database.azuredatabricks.net
  conn_db_name: your_database
  workspace_host: https://your-workspace.azuredatabricks.net

# Prompts (customize as needed)
prompts:
  planner_prompt: "..."
  topic_classifier_prompt: "..."
  replanner_prompt: "..."
  synthesis_prompt: "..."
```

### Environment Variables

```bash
# Optional - for tuning
export LOG_LEVEL=INFO
export DB_POOL_MIN_SIZE=2
export DB_POOL_MAX_SIZE=10
export DB_TOKEN_CACHE_MINUTES=50
```

---

## 💻 Usage

### Python API

```python
from agentv1_7 import PlannerResponsesAgent
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentRequestItem

# Initialize agent
agent = PlannerResponsesAgent(lakebase_config={...})

# Create request
request = ResponsesAgentRequest(
    input=[
        ResponsesAgentRequestItem(
            role="user",
            content="Who owns customer_analytics?"
        )
    ],
    custom_inputs={"thread_id": "unique-thread-id"}  # Optional
)

# Get response
response = agent.predict(request)
print(response.output[0]["text"])

# Get thread_id for continuation
thread_id = response.custom_outputs["thread_id"]
```

### Streaming API

```python
# Stream responses
for event in agent.predict_stream(request):
    if event.type == "response.output_item.done":
        print(event.item["text"])
```

### REST API (Model Serving)

```bash
curl -X POST https://your-workspace.azuredatabricks.net/serving-endpoints/agent/invocations \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": [
      {"role": "user", "content": "Who owns customer_analytics?"}
    ],
    "custom_inputs": {
      "thread_id": "550e8400-e29b-41d4-a716-446655440000"
    }
  }'
```

### Conversation Continuation

```python
# First query
request1 = ResponsesAgentRequest(
    input=[ResponsesAgentRequestItem(role="user", content="Who owns wells_master?")]
)
response1 = agent.predict(request1)
thread_id = response1.custom_outputs["thread_id"]

# Follow-up (same context)
request2 = ResponsesAgentRequest(
    input=[ResponsesAgentRequestItem(role="user", content="What about field_operations_events?")],
    custom_inputs={"thread_id": thread_id}
)
response2 = agent.predict(request2)
# Agent knows you're asking about another table in the same context
```

---

## 🏗️ Project Structure

```
intelligent-planning-agent/
│
├── agentv1_7.py                    # Main agent implementation
│   ├── Pydantic models (Plan, PlanStep, Classification)
│   ├── LangGraph workflow nodes
│   └── MLflow ResponsesAgent integration
│
├── lb_state_manager.py             # PostgreSQL state management
│   ├── OAuth token rotation
│   ├── Connection pooling
│   └── Checkpoint initialization
│
├── message_converters.py           # Message normalization
│   └── Handles checkpoint deserialization
│
├── config.yaml                     # Configuration file
│
├── notebooks/
│   └── Vector_Search_Setup.py     # Data pipeline for indices
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── LICENSE                         # Apache 2.0
```

---

## 🎨 Customization

### Adding New Vector Search Domains

1. **Create source table** with domain-specific fields
```sql
CREATE TABLE metadata_custom_source AS
SELECT 
  schema_name,
  max(CASE WHEN tag_name = 'custom_field' THEN tag_value END) AS field,
  ...
FROM tags
GROUP BY schema_name;
```

2. **Create vector index**
```sql
CREATE VECTOR SEARCH INDEX catalog.schema.metadata_custom_idx
ON TABLE catalog.schema.metadata_custom_source (document_text)
USING ENDPOINT embedding_endpoint;
```

3. **Register in config.yaml**
```yaml
tools:
  vector_search_custom_index: catalog.schema.metadata_custom_idx
```

4. **Add tool description** in prompts section

### Customizing Prompts

All prompts are externalized in `config.yaml`:
- `planner_prompt` - Planning instructions
- `topic_classifier_prompt` - Topic classification
- `replanner_prompt` - Replanning logic
- `synthesis_prompt` - Response generation

Modify these without code changes for rapid iteration.

### Integrating New Tools

```python
from databricks_langchain import UCFunctionToolkit

# Add Unity Catalog functions
UC_TOOL_NAMES = [
    "catalog.schema.my_custom_function"
]

uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
tools.extend(uc_toolkit.tools)
```

---

## 🔍 How It Works

### 1. Topic Classification

Before planning, the agent determines if the query is:
- **New Topic**: Fresh conversation, no prior context needed
- **Continuation**: References previous messages, includes history

```python
Classification(
    is_new_topic=False,
    reasoning="User said 'what about', referencing previous table query",
    confidence=0.95
)
```

### 2. Planning

Agent creates a structured plan with:
- Action type (tool_call, genie_query, synthesize)
- Tool selection (which domain index to use)
- Query transformation (optimize for vector search)
- Reasoning (why this step is needed)

### 3. Execution

Steps execute sequentially:
- Vector Search tools query domain-specific indices
- Genie agent handles analytical queries (costs, usage)
- Results stored for synthesis

### 4. Synthesis

Final LLM call combines all execution results into a coherent, natural language response.

### 5. State Persistence

Every state transition is checkpointed to PostgreSQL:
- Full message history
- Execution plans and results
- Classification outcomes
- Error states for replanning

---

## 📊 Performance

### Typical Latency

| Query Complexity | Average Time |
|-----------------|--------------|
| Simple (1 tool) | 4-6 seconds |
| Complex (3 tools) | 8-12 seconds |
| With replanning | +3-5 seconds per replan |

### Scaling

- **Concurrent users**: Supports horizontal scaling via Model Serving
- **Connection pooling**: 1-10 connections per replica
- **Token caching**: 50-minute OAuth token cache reduces API calls
- **Vector search**: Millisecond latency for semantic retrieval

---

## 🐛 Troubleshooting

### Common Issues

**Issue: "Tool not found" errors**
- Check tool names in config.yaml match registered tools
- Verify vector indices are created and synced
- Review planner prompt for correct tool names

**Issue: Empty search results**
- Verify vector indices have data
- Check query transformation in planner prompt
- Test queries directly against indices

**Issue: PostgreSQL connection failures**
- Verify Lakebase instance is running
- Check connection parameters in config
- Ensure service principal has database permissions
- Verify token cache is working (check logs)

**Issue: Conversations not persisting**
- Verify checkpoint tables exist
- Check thread_id is being passed correctly
- Review state_manager logs for errors

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Logs include:
- Planning steps and reasoning
- Tool invocations and results
- State transitions
- PostgreSQL operations

---

## 🤝 Contributing

We welcome contributions! Areas of interest:

- **Parallel execution** - Execute independent plan steps concurrently
- **Plan caching** - Cache and reuse plans for similar queries
- **Additional domains** - New vector search specializations
- **Tool integrations** - More UC functions, external APIs
- **UI improvements** - Better visualization of planning process

### Development Setup

```bash
# Clone repo
git clone https://github.com/your-org/intelligent-planning-agent.git

# Install in editable mode
pip install -e .

# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
isort .
```

---

## 🎓 Learn More

### Blog Posts
- [Building a Planning-First AI Agent]

### Related Projects
- [LangGraph](https://github.com/langchain-ai/langgraph) - Workflow orchestration framework
- [Databricks Vector Search](https://docs.databricks.com/en/generative-ai/vector-search.html) - Native vector search
- [MLflow](https://github.com/mlflow/mlflow) - Model serving and tracking

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built with:
- [LangGraph](https://langchain-ai.github.io/langgraph/) for workflow orchestration
- [Databricks](https://databricks.com) for the platform
- [MLflow](https://mlflow.org) for model serving
- [Pydantic](https://docs.pydantic.dev/) for structured data

---

## 🗺️ Roadmap

- [ ] Parallel step execution for independent operations
- [ ] Plan caching and reuse for similar queries
- [ ] Streaming synthesis (partial results as they arrive)
- [ ] Multi-modal support (images, documents)
- [ ] Custom embedding models per domain
- [ ] Real-time vector index updates
- [ ] Advanced error recovery strategies
- [ ] UI dashboard for conversation analytics

---

**Star ⭐ this repo if you find it useful!**

Built with ❤️ for the data community
