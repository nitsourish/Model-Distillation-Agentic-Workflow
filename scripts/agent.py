"""
Agent definitions for loan prediction application.
"""

import re
import operator
from typing import TypedDict, Annotated

from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


class LoanSimulatorAgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class LoanSimulatorAgent(StateGraph[LoanSimulatorAgentState]): 
    def __init__(self, model, tools, system_prompt):
        self.model = model
        self.tools = tools
        self.system_prompt = system_prompt
        
        self.model_with_tools = self.model.bind_tools(list(tools.values()))

        super().__init__(LoanSimulatorAgentState)
        self.add_node("loan_simulator_llm", self.call_llm)
        self.add_node("loan_simulator_tools", self.call_tools)
        
        self.set_entry_point("loan_simulator_llm")
        self.add_conditional_edges(
            "loan_simulator_llm",
            self.should_call_tools,
            {"tools": "loan_simulator_tools", "end": END}
        )
        self.add_conditional_edges(
            "loan_simulator_tools", 
            lambda state: "continue",
            {"continue": "loan_simulator_llm"}
        )
        
        self.agent_graph = self.compile(checkpointer=MemorySaver())

    def call_llm(self, state: LoanSimulatorAgentState):
        messages = state["messages"]
        if self.system_prompt:
            messages = [SystemMessage(content=self.system_prompt)] + messages
        result = self.model_with_tools.invoke(messages)
        return {"messages": [result]}

    def should_call_tools(self, state: LoanSimulatorAgentState):
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and len(last_message.tool_calls) > 0:
            return "tools"
        return "end"

    def call_tools(self, state: LoanSimulatorAgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []

        for tool in tool_calls:
            if tool["name"] not in self.tools:
                result = f"Invalid tool '{tool['name']}'"
            else:
                result = self.tools[tool["name"]].invoke(tool["args"])

            results.append(ToolMessage(
                tool_call_id=tool['id'], 
                name=tool['name'], 
                content=str(result)
            ))

        return {"messages": results}


class PrivacyAwareLoanAgent:
    def __init__(self, model, tools):
        self.current_customer_id = None
        self.memory = MemorySaver()
        
        self.system_prompt = """You are a professional loan analysis chatbot.

WORKFLOW:
1. Call 'load_data()' once at session start
2. Use tools:
   - 'get_customer_information(customer_id)' for customer details
   - 'load_predict_loan_approval_whatif(customer_id, scenario_description)' for what-if scenarios (includes loan amounts)
   - 'get_max_loan_amount(customer_id)' for current customer loan amount
   - 'load_predict_loan_amount_whatif(customer_id, scenario_description)' for detailed loan amount what-if

WHAT-IF ANALYSIS:
- Use natural language descriptions: "customer has masters degree and is married"
- Tools will intelligently parse and apply modifications
- Always use whatif tools for hypothetical scenarios

CONTEXT AWARENESS:
- Remember previous scenario modifications within the same customer session
- The whatif tools automatically include loan amount information in their results
- When asked about "loan amount in this scenario" or "maximum loan amount" after a what-if analysis, 
  look for the "scenario_max_loan_amount" field in the previous what-if result
- If no previous scenario exists, use 'get_max_loan_amount(customer_id)' for current customer data
- For follow-up questions about scenarios, re-run the whatif tool with the same scenario description

JSON OUTPUT FORMAT:
{
  "customer_id": 123,
  "scenario": "customer has masters degree",
  "modifications_applied": ["education=Masters"],
  "new_loan_approval_probability": 0.72,
  "baseline_loan_approval_probability": 0.49,
  "probability_change": "↗️ +23.0%",
  "approved": true
}

RULES:
- ALL responses MUST be in valid JSON format only, no additional text
- Remember customer context and scenario modifications for follow-up questions
- Privacy: clear context when switching customers
- Return ONLY the JSON, nothing else
"""
        
        self.agent = LoanSimulatorAgent(model, tools, self.system_prompt)
    
    def chat(self, user_message):
        customer_id_match = re.search(r'customer\s+(\d+)', user_message, re.IGNORECASE)
        mentioned_customer_id = None
        
        if customer_id_match:
            mentioned_customer_id = int(customer_id_match.group(1))
            
        if not mentioned_customer_id and self.current_customer_id:
            mentioned_customer_id = self.current_customer_id
            session_id = f"customer_{mentioned_customer_id}"
        elif mentioned_customer_id:
            # Use consistent session ID for the same customer to maintain context
            session_id = f"customer_{mentioned_customer_id}"
            self.current_customer_id = mentioned_customer_id
        else:
            session_id = "default_session"
        
        result = self.agent.agent_graph.invoke(
            {"messages": [HumanMessage(content=user_message)]},
            config={"configurable": {"thread_id": session_id}}
        )
        
        final_message = result["messages"][-1]
        return final_message.content


def create_privacy_aware_agent(model, tools):
    """Create a privacy-aware loan agent with the given model and tools."""
    return PrivacyAwareLoanAgent(model, tools)
