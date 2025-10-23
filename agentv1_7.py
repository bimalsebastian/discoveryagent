import functools
import json
import logging
import os
import uuid
from typing import Any, Generator, Literal, Optional, Dict, List, Union, Sequence, Annotated, TypedDict
from enum import Enum
import mlflow
from databricks.sdk import WorkspaceClient
from databricks_langchain import (
    ChatDatabricks,
    UCFunctionToolkit,
    VectorSearchRetrieverTool,
)
from langchain_core.language_models import LanguageModelLike
from langchain_core.tools import BaseTool 
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolMessage,
)
from databricks_langchain.genie import GenieAgent
from langchain_core.runnables import RunnableLambda, RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.postgres import PostgresSaver
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
)
from mlflow.models import ModelConfig
from pydantic import BaseModel, Field

# Import utilities 
from lb_state_manager import DatabricksStateManager
from message_converters import MessageNormalizer
from functools import wraps

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

config = ModelConfig(development_config="config.yaml").to_dict()
mlflow.set_tracking_uri("databricks")

###################################################
## Configuration
###################################################

LLM_ENDPOINT_NAME = config['llm_endpoint']
GENIE_SPACE_ID = config['tools']['genie_space_id'] 
GENIE_AGENT_NAME = config['tools']['genie_agent_name']
GENIE_AGENT_DESCRIPTION = config['prompts']['genie_system_prompt']
# TOOLS_SYSTEM_PROMPT = config['prompts']['tools_system_prompt']

# Prompts for planner architecture
TOPIC_CLASSIFIER_PROMPT = config['prompts']['topic_classifier_prompt']
PLANNER_PROMPT = config['prompts']['planner_prompt']
REPLANNER_PROMPT = config['prompts']['replanner_prompt']

###################################################
## Initialize LLM and Tools
###################################################

llm = ChatDatabricks(endpoint=LLM_ENDPOINT_NAME)

# Initialize tools
tools = []
UC_TOOL_NAMES: list[str] = [] 
if UC_TOOL_NAMES:
    uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
    tools.extend(uc_toolkit.tools)

# Vector search tools
VECTOR_SEARCH_TOOLS = [
    VectorSearchRetrieverTool(
        index_name=config['tools']["vector_search_identity_index"],
        tool_name="identity_search",
        tool_description=config["prompts"]["vector_search_identity_tool_desc"]
    ),
    VectorSearchRetrieverTool(
        index_name=config["tools"]["vector_search_business_index"],
        tool_name="business_search",
        tool_description=config["prompts"]["vector_search_business_tool_desc"]
    ),
    VectorSearchRetrieverTool(
        index_name=config["tools"]["vector_search_technical_index"],
        tool_name="technical_search",
        tool_description=config["prompts"]["vector_search_technical_tool_desc"]
    ),
    VectorSearchRetrieverTool(
        index_name=config["tools"]["vector_search_governance_index"],
        tool_name="governance_search",
        tool_description=config["prompts"]["vector_search_governance_tool_desc"]
    ),
]

tools.extend(VECTOR_SEARCH_TOOLS)

# Genie Agent
genie_agent = GenieAgent(
    genie_space_id=GENIE_SPACE_ID,
    genie_agent_name=GENIE_AGENT_NAME,
    description=GENIE_AGENT_DESCRIPTION
)

###################################################
## Pydantic Models for Planning
###################################################

class PlanStep(BaseModel):
    """A single step in the execution plan."""
    step_number: int = Field(description="Sequential step number")
    action: Literal["tool_call", "genie_query", "synthesize"] = Field(
        description="Type of action to perform"
    )
    tool_name: Optional[str] = Field(
        default=None, 
        description="Name of tool to call (for tool_call action)"
    )
    query: Optional[str] = Field(
        default=None,
        description="Query or input for the action"
    )
    reasoning: str = Field(description="Why this step is needed")


class Plan(BaseModel):
    """Complete execution plan."""
    steps: List[PlanStep] = Field(description="Ordered list of steps to execute")
    overall_strategy: str = Field(description="High-level strategy for answering the query")


class Classification(BaseModel):
    """Classification result for query context."""
    is_new_topic: bool = Field(description="True if query is unrelated to conversation history")
    reasoning: str = Field(description="Explanation for the classification")
    confidence: float = Field(description="Confidence score between 0 and 1")

class StateNormalizer:
    """Handles state object reconstruction from PostgreSQL checkpoints."""
    
    @staticmethod
    def normalize_state(state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reconstruct Pydantic objects from checkpoint dicts.
        
        PostgreSQL checkpointing serializes Pydantic objects as dicts.
        This reconstructs them back to proper Pydantic objects for type safety.
        
        Args:
            state: AgentState dict (may contain serialized Pydantic objects)
            
        Returns:
            State with Pydantic objects reconstructed
        """
        
        # Reconstruct Classification object
        if "classification" in state and state["classification"] is not None:
            if isinstance(state["classification"], dict):
                state["classification"] = Classification(**state["classification"])
        
        # Reconstruct Plan object (with nested PlanStep objects)
        if "plan" in state and state["plan"] is not None:
            if isinstance(state["plan"], dict):
                plan_dict = state["plan"].copy()
                
                # Reconstruct nested PlanStep objects
                if "steps" in plan_dict and plan_dict["steps"]:
                    plan_dict["steps"] = [
                        PlanStep(**step) if isinstance(step, dict) else step
                        for step in plan_dict["steps"]
                    ]
                
                state["plan"] = Plan(**plan_dict)
        
        return state
# Convenience function for the most common use case

def normalize_state_inputs(func):
    """
    Decorator to normalize state before node execution.
    
    Ensures Pydantic objects are reconstructed from checkpoint dicts.
    Apply to all LangGraph node functions that read state.
    
    Usage:
        @normalize_state_inputs
        def my_node(state: AgentState) -> AgentState:
            # State is already normalized here
            ...
    """
    @wraps(func)
    def wrapper(state):
        state = StateNormalizer.normalize_state(state)
        return func(state)
    return wrapper


###################################################
## State Schema
###################################################

class AgentState(TypedDict):
    """State for the planner-based agent."""
    messages: Annotated[List[BaseMessage], add_messages]
    classification: Optional[Classification]
    context_messages: List[BaseMessage]  # Filtered messages for planning
    plan: Optional[Plan]
    current_step: int
    execution_results: Dict[int, Any]  # step_number -> result
    replan_count: int
    final_response: Optional[str]
    error: Optional[str]


###################################################
## Helper Functions
###################################################

def build_tool_descriptions() -> str:
    """Build formatted descriptions of all available tools."""
    descriptions = []
    
    # Vector search tools
    for tool in VECTOR_SEARCH_TOOLS:
        descriptions.append(f"- {tool.name}: {tool.description}")
    
    # UC tools
    for tool in tools:
        if tool.name not in [t.name for t in VECTOR_SEARCH_TOOLS]:
            descriptions.append(f"- {tool.name}: {tool.description}")
    
    # Genie agent
    descriptions.append(f"- {GENIE_AGENT_NAME}: {GENIE_AGENT_DESCRIPTION}")
    
    return "\n".join(descriptions)


###################################################
## Node Functions
###################################################
@normalize_state_inputs
def classify_topic(state: AgentState) -> AgentState:
    """Classify if the current query is a new topic or continuation."""
    messages = state.get("messages", [])
    
    # Normalize messages from checkpoint (handle complex content structures)
    messages = MessageNormalizer.normalize_messages(messages)
    state["messages"] = messages
    
    # If no history, it's definitely a new topic
    if len(messages) <= 1:
        state["classification"] = Classification(
            is_new_topic=True,
            reasoning="No conversation history exists",
            confidence=1.0
        )
        state["context_messages"] = [messages[-1]] if messages else []
        return state
    
    # Use LLM to classify
    current_query = messages[-1].content if messages[-1].content else ""
    
    # Build history summary from recent messages
    history_messages = messages[-6:-1] if len(messages) > 1 else []
    history_summary = "\n".join([
        f"{msg.type}: {msg.content[:200]}..." if len(msg.content) > 200 else f"{msg.type}: {msg.content}"
        for msg in history_messages
        if hasattr(msg, 'content') and msg.content
    ])
    
    if not history_summary:
        history_summary = "No previous conversation"
    
    try:
        classification_prompt = TOPIC_CLASSIFIER_PROMPT.format(
            history=history_summary,
            current_query=current_query
        )
        
        structured_llm = llm.with_structured_output(Classification)
        classification = structured_llm.invoke([HumanMessage(content=classification_prompt)])
        
        state["classification"] = classification
        
        # Prepare context messages
        if classification.is_new_topic:
            # Only use current query
            state["context_messages"] = [messages[-1]]
            logger.info(f"Classified as NEW TOPIC (confidence: {classification.confidence})")
        else:
            # Include relevant history (last 10 messages)
            state["context_messages"] = messages[-10:]
            logger.info(f"Classified as CONTINUATION (confidence: {classification.confidence})")
    except Exception as e:
        logger.error(f"Classification failed: {str(e)}", exc_info=True)
        # Default to new topic on error
        state["classification"] = Classification(
            is_new_topic=True,
            reasoning=f"Classification failed: {str(e)}",
            confidence=0.5
        )
        state["context_messages"] = [messages[-1]] if messages else []
    
    return state

@normalize_state_inputs
def create_plan(state: AgentState) -> AgentState:
    """Generate execution plan based on query and context."""
    context_messages = state.get("context_messages", [])
    if not context_messages:
        logger.error("No context messages available for planning")
        state["error"] = "No context messages available"
        return state
    
    current_query = context_messages[-1].content

    logger.info(f"=== PLANNING DEBUG ===")
    logger.info(f"Current query: {current_query}")
    logger.info(f"Context messages count: {len(context_messages)}")

    # Build context with tool descriptions
    tool_descriptions = build_tool_descriptions()
    logger.info(f"Tool descriptions length: {len(tool_descriptions)}")
    logger.info(f"Tool descriptions preview: {tool_descriptions[:200]}...")

    # Build context string
    context_str = "\n".join([
        f"{msg.type}: {msg.content}" 
        for msg in context_messages[:-1]
        if hasattr(msg, 'content') and msg.content
    ]) if len(context_messages) > 1 else "No previous context"
    
    logger.info(f"Context string: {context_str}")

    try:
        planning_prompt = PLANNER_PROMPT.format(
            tools=tool_descriptions,
            query=current_query,
            context=context_str
        )
        logger.info(f"Planning prompt length: {len(planning_prompt)}")
        logger.info(f"Planning prompt first 300 chars: {planning_prompt[:300]}")
    except KeyError as e:
        logger.error(f"Prompt formatting failed - missing key: {e}")
        state["error"] = f"Prompt formatting failed: {e}"
        return state
    
    try:
        if not planning_prompt or not planning_prompt.strip():
            logger.error(f"Planning prompt is empty! Tool descriptions length: {len(tool_descriptions)}, Query: {current_query}")
            state["error"] = "Planning prompt is empty"
            return state

        logger.debug(f"Planning prompt length: {len(planning_prompt)}")
        logger.debug(f"Planning prompt preview: {planning_prompt[:200]}...")

        structured_llm = llm.with_structured_output(Plan)
        plan = structured_llm.invoke([HumanMessage(content=planning_prompt)])
        
        state["plan"] = plan
        state["current_step"] = 0
        state["execution_results"] = {}
        state["replan_count"] = state.get("replan_count", 0)  # Initialize if not present
        
        logger.info(f"Generated plan with {len(plan.steps)} steps")
        logger.info(f"Strategy: {plan.overall_strategy}")
    except Exception as e:
        logger.error(f"Failed to generate plan: {str(e)}", exc_info=True)
        state["error"] = f"Planning failed: {str(e)}"
    
    return state

@normalize_state_inputs
def replan(state: AgentState) -> AgentState:
    """Replan based on execution failures."""
    state["replan_count"] = state.get("replan_count", 0) + 1
    
    if state["replan_count"] > 2:
        state["error"] = "Maximum replan attempts (2) exceeded"
        return state
    
    # Get context about what failed
    failed_step = 0
    if "current_step" in state:
        failed_step = state["current_step"]
    original_plan = state.get("plan")
    if not original_plan:
        logger.error("No plan available for replanning")
        state["error"] = "Cannot replan without an original plan"
        return state
    execution_results = state["execution_results"]
    error_info = state.get("error", "Unknown error")
    
    replanning_prompt = REPLANNER_PROMPT.format(
        original_plan=json.dumps(original_plan.model_dump(), indent=2),
        failed_step=failed_step,
        error=error_info,
        execution_results=json.dumps(execution_results, indent=2),
        tools=build_tool_descriptions()
    )
    
    structured_llm = llm.with_structured_output(Plan)
    new_plan = structured_llm.invoke([HumanMessage(content=replanning_prompt)])
    
    state["plan"] = new_plan
    state["current_step"] = 0
    state["error"] = None
    
    logger.info(f"Replanned (attempt {state['replan_count']}): {new_plan.overall_strategy}")
    
    return state

@normalize_state_inputs
def execute_step(state: AgentState) -> AgentState:
    """Execute a single step from the plan."""
    plan = state.get("plan")
    if not plan:
        logger.error("No plan found in state")
        state["error"] = "No execution plan available"
        return state
    
    current_step_idx = state.get("current_step", 0)
    
    if current_step_idx >= len(plan.steps):
        # All steps completed
        return state
    
    step = plan.steps[current_step_idx]
    logger.info(f"Executing step {step.step_number}: {step.action} - {step.reasoning}")
    
    try:
        result = None
        
        if step.action == "tool_call":
            # Find and execute the tool
            tool = next((t for t in tools if t.name == step.tool_name), None)
            if not tool:
                available_tools = [t.name for t in tools]
                error_msg = f"Tool '{step.tool_name}' not found. Available tools: {', '.join(available_tools)}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            logger.info(f"Invoking tool '{step.tool_name}' with query: {step.query[:100]}...")
            result = tool.invoke(step.query)
            logger.info(f"Tool '{step.tool_name}' returned result of length: {len(str(result))}")
            
        elif step.action == "genie_query":
            # Execute Genie agent
            logger.info(f"Invoking Genie agent with query: {step.query[:100]}...")
            genie_result = genie_agent.invoke({"messages": [HumanMessage(content=step.query)]})
            
            # Extract the response from genie
            if isinstance(genie_result, dict) and "messages" in genie_result:
                result = genie_result["messages"][-1].content
            else:
                result = str(genie_result)
            logger.info(f"Genie agent returned result of length: {len(str(result))}")
                
        elif step.action == "synthesize":
            # Synthesize results from previous steps
            result = "synthesis_placeholder"  # Will be handled in synthesize_response
            logger.info("Marked step for synthesis")
        
        # Store result
        if "execution_results" not in state:
            state["execution_results"] = {}
        state["execution_results"][step.step_number] = result
        state["current_step"] = current_step_idx + 1
        
        logger.info(f"Step {step.step_number} completed successfully")
        
    except Exception as e:
        logger.error(f"Step {step.step_number} failed: {str(e)}", exc_info=True)
        state["error"] = f"Step {step.step_number} failed: {str(e)}"
    
    return state

@normalize_state_inputs
def synthesize_response(state: AgentState) -> AgentState:
    """Synthesize final response from execution results."""
    plan = state["plan"]
    execution_results = state["execution_results"]
    current_query = state["context_messages"][-1].content
    
    # Build synthesis context
    results_summary = []
    for step in plan.steps:
        if step.step_number in execution_results:
            result = execution_results[step.step_number]
            # Truncate long results but keep meaningful information
            result_str = str(result)
            if len(result_str) > 1000:
                result_str = result_str[:1000] + "... (truncated)"
            results_summary.append(
                f"Step {step.step_number} ({step.action}): {step.reasoning}\n"
                f"Result: {result_str}"
            )
    
    # Check if synthesis prompt exists in config, otherwise use default
    synthesis_template = config['prompts']['synthesis_prompt']
    synthesis_prompt = synthesis_template.format(
        query=current_query,
        results="\n\n".join(results_summary),
        strategy=plan.overall_strategy
    )

    
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    state["final_response"] = response.content
    
    # Add the response as an AI message
    state["messages"].append(AIMessage(content=response.content))
    
    logger.info("Response synthesized successfully")
    
    return state

@normalize_state_inputs
def should_continue_execution(state: AgentState) -> str:
    """Determine next step in the workflow."""

    if not state.get("plan"):
        logger.error("No plan exists, cannot continue execution")
        return "end"
    # Check for errors
    if state.get("error"):
        # Only replan if we have a valid plan and haven't exceeded retries
        if state.get("plan") and state.get("replan_count", 0) < 2:
            return "replan"
        else:
            return "end"
    
    # Check if all steps are executed
    plan = state.get("plan")
    current_step = state.get("current_step", 0)
    
    if plan and current_step < len(plan.steps):
        return "execute"
    else:
        return "synthesize"

@normalize_state_inputs
def should_replan_or_end(state: AgentState) -> str:
    """Determine if we should replan or end after error."""
    if state.get("replan_count", 0) < 2:
        return "replan"
    return "end"


###################################################
## Build LangGraph Workflow
###################################################

def create_planner_graph(checkpointer: PostgresSaver) -> StateGraph:
    """Create the planner-based workflow graph."""
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("classify", classify_topic)
    workflow.add_node("plan", create_plan)
    workflow.add_node("execute", execute_step)
    workflow.add_node("replan", replan)
    workflow.add_node("synthesize", synthesize_response)
    
    # Define edges
    workflow.set_entry_point("classify")
    workflow.add_edge("classify", "plan")
    workflow.add_edge("plan", "execute")
    
    # Conditional routing after execution
    workflow.add_conditional_edges(
        "execute",
        should_continue_execution,
        {
            "execute": "execute",  # Loop back for next step
            "synthesize": "synthesize",
            "replan": "replan",
            "end": END
        }
    )
    
    workflow.add_edge("replan", "execute")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile(checkpointer=checkpointer)


###################################################
## Main ResponsesAgent Implementation
###################################################

class PlannerResponsesAgent(ResponsesAgent):
    """Planner-based multi-agent using ResponsesAgent with PostgreSQL checkpointing."""
    
    def __init__(self, lakebase_config: dict[str, Any]):
        self.lakebase_config = lakebase_config
        self.workspace_client = WorkspaceClient()
        
        # Initialize state manager
        self.state_manager = DatabricksStateManager(lakebase_config=lakebase_config)
        
        mlflow.langchain.autolog()
    
    def _langchain_to_responses(self, message: BaseMessage) -> list[dict[str, Any]]:
        """Convert LangChain message to Responses output items."""
        message_dict = message.model_dump()
        role = message_dict["type"]
        output = []
        
        if role == "ai":
            # Handle text content
            content = message_dict.get("content", "")
            if content and isinstance(content, str):
                output.append(
                    self.create_text_output_item(
                        text=content,
                        id=message_dict.get("id") or str(uuid.uuid4()),
                    )
                )
            
            # Handle tool calls
            if tool_calls := message_dict.get("tool_calls"):
                output.extend([
                    self.create_function_call_item(
                        id=message_dict.get("id") or str(uuid.uuid4()),
                        call_id=tool_call["id"],
                        name=tool_call["name"],
                        arguments=json.dumps(tool_call["args"])
                    )
                    for tool_call in tool_calls
                ])
        
        elif role == "tool":
            output.append(
                self.create_function_call_output_item(
                    call_id=message_dict["tool_call_id"],
                    output=message_dict["content"],
                )
            )
        
        return output
    
    def _create_planning_status_message(self, plan: Plan) -> dict[str, Any]:
        """Create a user-friendly status message showing the plan."""
        
        # Create a conversational, high-level summary
        action_verbs = {
            "tool_call": "searching",
            "genie_query": "analyzing",
            "synthesize": "compiling"
        }
        
        # Count different action types
        search_steps = sum(1 for s in plan.steps if s.action == "tool_call")
        genie_steps = sum(1 for s in plan.steps if s.action == "genie_query")
        
        # Build friendly message
        plan_text = "💡 **I understand your question! Let me help you with that.**\n\n"
        plan_text += "I'm going to gather the information you need by checking a few different sources. "
        plan_text += "This should only take a moment...\n\n"
        plan_text += "🔄 **Starting now...**"
        
        if search_steps > 0:
            plan_text += f"I'll search through {search_steps} relevant data source{'s' if search_steps > 1 else ''} "
        if genie_steps > 0:
            plan_text += f"and analyze quantitative data "
        plan_text += "to give you a comprehensive answer.\n\n"
        
        plan_text += "🔄 **Working on it now...**"
        
        return self.create_text_output_item(
            text=plan_text,
            id=str(uuid.uuid4())
        )
    
    def _create_step_status_message(self, step: PlanStep, status: str) -> dict[str, Any]:
        """Create a user-friendly status message for step execution."""
        
        # Map actions to user-friendly messages
        action_messages = {
            "tool_call": {
                "identity_search": "🔍 Checking ownership and contact information...",
                "business_search": "📊 Reviewing business context and documentation...",
                "technical_search": "🔧 Looking up technical specifications...",
                "governance_search": "🛡️ Verifying governance and compliance details..."
            },
            "genie_query": "📈 Running data analysis...",
            "synthesize": "✨ Putting it all together..."
        }
        
        # Get appropriate message based on action type
        if step.action == "tool_call" and step.tool_name:
            status_text = action_messages["tool_call"].get(
                step.tool_name,
                f"🔍 Searching {step.tool_name.replace('_', ' ')}..."
            )
        elif step.action == "genie_query":
            status_text = action_messages["genie_query"]
        elif step.action == "synthesize":
            status_text = action_messages["synthesize"]
        else:
            status_text = f"⏳ Step {step.step_number}: {status}"
        
        return self.create_text_output_item(
            text=status_text,
            id=str(uuid.uuid4())
        )
    
    def predict(self, request: Union[ResponsesAgentRequest, Dict[str, Any]]) -> ResponsesAgentResponse:
        """Non-streaming prediction."""
        # Handle dict input
        if isinstance(request, dict):
            from mlflow.types.responses import ResponsesAgentRequestItem
            input_items = []
            for item in request.get("input", []):
                if isinstance(item, dict):
                    input_items.append(ResponsesAgentRequestItem(**item))
                else:
                    input_items.append(item)
            
            request = ResponsesAgentRequest(
                input=input_items,
                custom_inputs=request.get("custom_inputs", {})
            )
        
        ci = dict(request.custom_inputs or {})
        if "thread_id" not in ci:
            ci["thread_id"] = str(uuid.uuid4())
        request.custom_inputs = ci
        
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        
        # Include thread_id in custom outputs
        custom_outputs = {"thread_id": ci["thread_id"]}
        
        return ResponsesAgentResponse(output=outputs, custom_outputs=custom_outputs)
    
    def predict_stream(
        self,
        request: ResponsesAgentRequest,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        """Streaming prediction with plan visibility."""
        thread_id = (request.custom_inputs or {}).get("thread_id", str(uuid.uuid4()))
        # Track what we've already yielded
        plan_shown = False
        last_step_shown = -1
        classification_shown = False 
        # Convert input to LangChain messages
        input_messages = []
        for item in request.input:
            item_dict = item.model_dump() if hasattr(item, 'model_dump') else item
            role = item_dict.get("role", "user")
            content = item_dict.get("content", "")
            
            if role == "user":
                input_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                input_messages.append(AIMessage(content=content))
            elif role == "system":
                input_messages.append(SystemMessage(content=content))
        
        checkpoint_config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 50
        }
        
        with self.state_manager.get_connection() as conn:
            checkpointer = self.state_manager.create_checkpointer(conn)
            graph = create_planner_graph(checkpointer)
            
            # Track what we've already yielded
            plan_shown = False
            last_step_shown = -1
            
            try:
                for event in graph.stream(
                    {"messages": input_messages},
                    checkpoint_config,
                    stream_mode="values"
                ):
                    event = StateNormalizer.normalize_state(event)
                    # Show classification
                    if "classification" in event and event["classification"] and not classification_shown:
                        classification = event["classification"]
                        
                        # Create friendly classification message
                        if classification.is_new_topic:
                            message = "👋 Starting fresh with your question..."
                        else:
                            message = "🔗 Building on our previous conversation..."
                        
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=self.create_text_output_item(
                                text=message,
                                id=str(uuid.uuid4())
                            )
                        )
                        classification_shown = True
                    
                    # Show plan
                    if "plan" in event and event["plan"] and not plan_shown:
                        plan = event["plan"]
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=self._create_planning_status_message(plan)
                        )
                        plan_shown = True
                    
                    # Show step execution progress (only when step actually changes)
                    if "current_step" in event and "plan" in event:
                        current_step_idx = event["current_step"]
                        plan = event["plan"]
                        
                        # Only show if this is a NEW step (not a duplicate event)
                        if current_step_idx > last_step_shown and current_step_idx <= len(plan.steps) and current_step_idx > 0:
                            step = plan.steps[current_step_idx - 1]
                            yield ResponsesAgentStreamEvent(
                                type="response.output_item.done",
                                item=self._create_step_status_message(step, "Executing...")
                            )
                            last_step_shown = current_step_idx
                    
                    # Show replanning with friendly message
                    if "replan_count" in event and event.get("replan_count", 0) > 0:
                        replan_count = event["replan_count"]
                        
                        friendly_messages = {
                            1: "🔄 Let me try a different approach...",
                            2: "🔄 One more attempt with an alternative strategy..."
                        }
                        
                        message = friendly_messages.get(replan_count, f"🔄 Adjusting my approach (attempt {replan_count})...")
                        
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=self.create_text_output_item(
                                text=message,
                                id=str(uuid.uuid4())
                            )
                        )
                        plan_shown = False  # Reset to show new plan
                        classification_shown = False  # Reset classification
                    
                    # Return final response (only the latest AI message)
                    if "final_response" in event and event["final_response"]:
                        # Only return the latest AI message
                        messages = event.get("messages", [])
                        if messages and isinstance(messages[-1], AIMessage):
                            items = self._langchain_to_responses(messages[-1])
                            for item in items:
                                yield ResponsesAgentStreamEvent(
                                    type="response.output_item.done",
                                    item=item
                                )
                    
                    # Handle errors
                    if "error" in event and event["error"]:
                        yield ResponsesAgentStreamEvent(
                            type="response.output_item.done",
                            item=self.create_text_output_item(
                                text=f"❌ Error: {event['error']}",
                                id=str(uuid.uuid4())
                            )
                        )
                        
            except Exception as e:
                logger.error(f"Error during agent streaming: {e}", exc_info=True)
                yield ResponsesAgentStreamEvent(
                    type="response.output_item.done",
                    item=self.create_text_output_item(
                        text=f"❌ Fatal Error: {str(e)}",
                        id=str(uuid.uuid4())
                    )
                )
                raise


# ----- Export model -----
AGENT = PlannerResponsesAgent(config["lakebase_config"])
mlflow.models.set_model(AGENT)
