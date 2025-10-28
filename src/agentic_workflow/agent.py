"""
Agentic workflow using LangGraph to consume distilled models.
This module creates an agent that can use the distilled model for various tasks.
"""

from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for the agent workflow."""
    messages: Annotated[Sequence[BaseMessage], "The messages in the conversation"]
    context: Annotated[str, "Additional context for the agent"]
    current_task: Annotated[Optional[str], "Current task being processed"]
    results: Annotated[List[str], "Results from agent actions"]


class DistilledModelAgent:
    """Agent that uses a distilled model for inference."""
    
    def __init__(
        self,
        model_path: str,
        device: str = "cpu",
        max_length: int = 100,
        temperature: float = 0.7,
    ):
        """
        Initialize the distilled model agent.
        
        Args:
            model_path: Path to the distilled model
            device: Device to run the model on ('cpu' or 'cuda')
            max_length: Maximum length for generated text
            temperature: Temperature for text generation
        """
        self.model_path = Path(model_path)
        self.device = device
        self.max_length = max_length
        self.temperature = temperature
        
        logger.info(f"Loading distilled model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create text generation pipeline
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
            max_length=max_length,
            temperature=temperature,
        )
        
        logger.info("Distilled model loaded successfully")
    
    def generate(self, prompt: str) -> str:
        """
        Generate text using the distilled model.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Generated text
        """
        try:
            result = self.generator(
                prompt,
                max_length=self.max_length,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            generated_text = result[0]['generated_text']
            # Remove the prompt from the generated text
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            return generated_text
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error: {str(e)}"
    
    def answer_question(self, question: str, context: Optional[str] = None) -> str:
        """
        Answer a question using the distilled model.
        
        Args:
            question: Question to answer
            context: Optional context for the question
            
        Returns:
            Answer to the question
        """
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        return self.generate(prompt)


class AgenticWorkflow:
    """LangGraph-based agentic workflow using distilled models."""
    
    def __init__(self, distilled_model_path: str):
        """
        Initialize the agentic workflow.
        
        Args:
            distilled_model_path: Path to the distilled model
        """
        self.agent = DistilledModelAgent(distilled_model_path)
        self.graph = self._create_graph()
        logger.info("Agentic workflow initialized")
    
    def _create_graph(self) -> StateGraph:
        """
        Create the LangGraph workflow.
        
        Returns:
            Compiled StateGraph
        """
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("process_input", self._process_input_node)
        workflow.add_node("generate_response", self._generate_response_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define edges
        workflow.set_entry_point("process_input")
        workflow.add_edge("process_input", "generate_response")
        workflow.add_edge("generate_response", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _process_input_node(self, state: AgentState) -> AgentState:
        """
        Process input messages.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state
        """
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                state["current_task"] = last_message.content
                logger.info(f"Processing task: {state['current_task']}")
        return state
    
    def _generate_response_node(self, state: AgentState) -> AgentState:
        """
        Generate response using the distilled model.
        
        Args:
            state: Current agent state
            
        Returns:
            Updated state with generated response
        """
        current_task = state.get("current_task", "")
        context = state.get("context", "")
        
        if current_task:
            # Generate response using distilled model
            response = self.agent.answer_question(current_task, context)
            
            # Add response to messages
            messages = list(state.get("messages", []))
            messages.append(AIMessage(content=response))
            state["messages"] = messages
            
            # Add to results
            results = list(state.get("results", []))
            results.append(response)
            state["results"] = results
            
            logger.info(f"Generated response: {response[:100]}...")
        
        return state
    
    def _finalize_node(self, state: AgentState) -> AgentState:
        """
        Finalize the workflow.
        
        Args:
            state: Current agent state
            
        Returns:
            Final state
        """
        logger.info("Workflow completed")
        return state
    
    def run(
        self,
        query: str,
        context: Optional[str] = None,
        chat_history: Optional[List[BaseMessage]] = None
    ) -> Dict:
        """
        Run the agentic workflow.
        
        Args:
            query: User query
            context: Optional context
            chat_history: Optional chat history
            
        Returns:
            Workflow results
        """
        # Initialize state
        messages = chat_history or []
        messages.append(HumanMessage(content=query))
        
        initial_state = {
            "messages": messages,
            "context": context or "",
            "current_task": None,
            "results": []
        }
        
        # Run the graph
        logger.info(f"Running workflow for query: {query}")
        final_state = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "response": final_state["results"][-1] if final_state["results"] else "",
            "messages": final_state["messages"],
            "all_results": final_state["results"]
        }
    
    def run_batch(
        self,
        queries: List[str],
        contexts: Optional[List[str]] = None
    ) -> List[Dict]:
        """
        Run the workflow on multiple queries.
        
        Args:
            queries: List of queries
            contexts: Optional list of contexts
            
        Returns:
            List of results
        """
        results = []
        contexts = contexts or [None] * len(queries)
        
        for query, context in zip(queries, contexts):
            result = self.run(query, context)
            results.append(result)
        
        return results


class MultiAgentWorkflow:
    """Multi-agent workflow for complex tasks."""
    
    def __init__(self, distilled_model_path: str):
        """
        Initialize multi-agent workflow.
        
        Args:
            distilled_model_path: Path to the distilled model
        """
        self.agent = DistilledModelAgent(distilled_model_path)
        self.graph = self._create_graph()
        logger.info("Multi-agent workflow initialized")
    
    def _create_graph(self) -> StateGraph:
        """Create a more complex multi-agent graph."""
        workflow = StateGraph(AgentState)
        
        # Add nodes for different agent roles
        workflow.add_node("router", self._router_node)
        workflow.add_node("qa_agent", self._qa_agent_node)
        workflow.add_node("summarizer", self._summarizer_node)
        workflow.add_node("validator", self._validator_node)
        
        # Define edges
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            self._route_decision,
            {
                "qa": "qa_agent",
                "summarize": "summarizer",
                "validate": "validator"
            }
        )
        workflow.add_edge("qa_agent", "validator")
        workflow.add_edge("summarizer", "validator")
        workflow.add_edge("validator", END)
        
        return workflow.compile()
    
    def _router_node(self, state: AgentState) -> AgentState:
        """Route to appropriate agent based on task."""
        messages = state.get("messages", [])
        if messages:
            last_message = messages[-1]
            if isinstance(last_message, HumanMessage):
                content = last_message.content.lower()
                if "summarize" in content or "summary" in content:
                    state["current_task"] = "summarize"
                elif "validate" in content or "check" in content:
                    state["current_task"] = "validate"
                else:
                    state["current_task"] = "qa"
        return state
    
    def _route_decision(self, state: AgentState) -> str:
        """Decide which agent to route to."""
        task = state.get("current_task", "qa")
        return task
    
    def _qa_agent_node(self, state: AgentState) -> AgentState:
        """QA agent node."""
        messages = state.get("messages", [])
        if messages:
            query = messages[-1].content
            response = self.agent.answer_question(query, state.get("context"))
            messages.append(AIMessage(content=response))
            state["messages"] = messages
        return state
    
    def _summarizer_node(self, state: AgentState) -> AgentState:
        """Summarizer agent node."""
        messages = state.get("messages", [])
        if messages:
            content = messages[-1].content
            prompt = f"Summarize the following: {content}"
            response = self.agent.generate(prompt)
            messages.append(AIMessage(content=response))
            state["messages"] = messages
        return state
    
    def _validator_node(self, state: AgentState) -> AgentState:
        """Validator agent node."""
        results = list(state.get("results", []))
        messages = state.get("messages", [])
        
        if messages and len(messages) >= 2:
            results.append(messages[-1].content)
        
        state["results"] = results
        return state
    
    def run(self, query: str, context: Optional[str] = None) -> Dict:
        """Run the multi-agent workflow."""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "context": context or "",
            "current_task": None,
            "results": []
        }
        
        final_state = self.graph.invoke(initial_state)
        
        return {
            "query": query,
            "response": final_state["results"][-1] if final_state["results"] else "",
            "task_type": final_state.get("current_task", "unknown"),
            "messages": final_state["messages"]
        }
