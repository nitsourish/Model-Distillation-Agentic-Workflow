"""
Streamlit app for What-If Analysis Business Scenario simulator.

A context-aware, privacy-preserving agentic workflow for loan approval using LangChain/LangGraph.
"""

import streamlit as st
import json
import sys
import os

# Add the parent directory to Python path for prediction_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import setup_azure_openai_model
from tools import get_all_tools
from agent import create_privacy_aware_agent


def main():
    st.set_page_config(
        page_title="🏦 Privacy-Aware Loan Agent",
        page_icon="🏦",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Header
    st.title("🏦 What-If Analysis: Business Scenario Simulator")
    st.markdown("""
    **Context-aware, privacy-preserving agentic workflow for loan approval using LangChain/LangGraph.**
    
    ### Key Features
    - 🔒 **Privacy Protection**: Context isolation between customers
    - 🧠 **Context Awareness**: Memory within customer sessions  
    - 🎯 **What-if Analysis**: Real ML predictions for scenarios
    - 📊 **JSON Responses**: Structured, concise outputs
    """)

    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Initialize session state
        if 'agent' not in st.session_state:
            with st.spinner("🚀 Initializing AI Agent..."):
                try:
                    # Setup model and tools
                    model = setup_azure_openai_model()
                    tools = get_all_tools(model)
                    
                    # Create agent
                    st.session_state.agent = create_privacy_aware_agent(model, tools)
                    
                    # Load data automatically
                    st.session_state.agent.chat("Load the data")
                    
                    st.success("✅ Agent initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to initialize agent: {str(e)}")
                    st.stop()
        
        st.success("🤖 Agent Status: Ready")
        
        # Debug section
        with st.expander("🔍 Debug Tools", expanded=False):
            if st.button("🔄 Reset Agent & Clear Memory"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        # Customer ID input
        st.subheader("👤 Customer Selection")
        customer_id = st.number_input(
            "Customer ID", 
            min_value=1, 
            value=1, 
            help="Enter a customer ID to analyze. This will be automatically included in your queries."
        )
        
        # Visual confirmation
        st.success(f"🎯 Selected: Customer **{customer_id}**")
        
        # Quick action buttons
        st.subheader("⚡ Quick Actions")
        if st.button("📋 Get Customer Info", use_container_width=True):
            st.session_state.quick_query = f"Get detailed information for customer {customer_id}"
        
        if st.button("🎯 Loan Probability", use_container_width=True):
            st.session_state.quick_query = f"What is the loan approval probability for customer {customer_id}?"
        
        if st.button("💰 Loan Amount", use_container_width=True):
            st.session_state.quick_query = f"What is the predicted loan amount for customer {customer_id}?"

    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat with Agent")
        
        # Current customer context
        st.info(f"🎯 **Current Context:** Customer ID {customer_id}")
        
        # Chat input
        if 'quick_query' in st.session_state:
            default_query = st.session_state.quick_query
            del st.session_state.quick_query
        else:
            default_query = ""
            
        user_input = st.text_area(
            "Enter your question or scenario:",
            value=default_query,
            height=100,
            placeholder=f"Ask about customer {customer_id} or use scenarios like: 'What are the details and loan approval probability?' or 'What if they had a Masters degree?'"
        )
        
        col_send, col_clear = st.columns([1, 1])
        with col_send:
            send_button = st.button("🚀 Send", use_container_width=True, type="primary")
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Process user input
        if send_button and user_input.strip():
            with st.spinner("🤔 Agent is thinking..."):
                try:
                    # Auto-add customer context if not explicitly mentioned
                    enhanced_query = user_input
                    if not any(word in user_input.lower() for word in ['customer', 'id']) and not user_input.strip().lower().startswith(('load', 'get', 'what if customer')):
                        enhanced_query = f"For customer {customer_id}: {user_input}"
                    
                    response = st.session_state.agent.chat(enhanced_query)
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "user": user_input,  # Store original user input
                        "enhanced_query": enhanced_query,  # Store enhanced query
                        "agent": response,
                        "customer_id": customer_id
                    })
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        # Display chat history
        if st.session_state.chat_history:
            st.header("📈 Chat History")
            
            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.expander(f"💬 Query {len(st.session_state.chat_history) - i}: Customer {chat['customer_id']}", expanded=(i == 0)):
                    st.markdown(f"**👤 You:** {chat['user']}")
                    
                    # Show enhanced query if different from original
                    if 'enhanced_query' in chat and chat['enhanced_query'] != chat['user']:
                        st.markdown(f"**🔧 Enhanced Query:** {chat['enhanced_query']}")
                    
                    # Try to parse and display JSON response nicely
                    try:
                        parsed_response = json.loads(chat['agent'])
                        st.markdown("**🤖 Agent Response:**")
                        st.json(parsed_response)
                        
                        # Extract key metrics if available
                        if 'probability_change' in parsed_response:
                            col_metric1, col_metric2, col_metric3 = st.columns(3)
                            
                            with col_metric1:
                                if 'new_loan_approval_probability' in parsed_response:
                                    st.metric(
                                        "Approval Probability", 
                                        f"{parsed_response['new_loan_approval_probability']:.1%}"
                                    )
                            
                            with col_metric2:
                                st.metric("Change", parsed_response['probability_change'])
                            
                            with col_metric3:
                                if 'approved' in parsed_response:
                                    status = "✅ Approved" if parsed_response['approved'] else "❌ Rejected"
                                    st.metric("Status", status)
                        
                    except json.JSONDecodeError:
                        st.markdown(f"**🤖 Agent:** {chat['agent']}")
    
    with col2:
        st.header("📊 Example Scenarios")
        
        example_scenarios = [
            {
                "title": "🎓 Education Impact",
                "query": "What if they had a Masters degree?",
                "description": "See how higher education affects loan approval"
            },
            {
                "title": "💰 Income Boost", 
                "query": "What if they had capital gain of 15000?",
                "description": "Analyze the impact of higher income"
            },
            {
                "title": "👔 Career Change",
                "query": "What if they worked in Exec-managerial role?",
                "description": "Effect of management position"
            },
            {
                "title": "🚀 Combined Scenario",
                "query": "What if they had ALL these: Masters degree, capital-gain of 10000, and worked 45 hours per week in Exec-managerial?",
                "description": "Multiple improvements combined"
            },
            {
                "title": "💳 Loan Amount",
                "query": "What is the maximum approved loan amount?",
                "description": "Get loan amount prediction"
            },
            {
                "title": "📋 Full Details",
                "query": "What are the details including loan approval probability?",
                "description": "Complete customer profile and current status"
            }
        ]
        
        for scenario in example_scenarios:
            with st.expander(scenario["title"]):
                st.markdown(scenario["description"])
                if st.button(f"Try this scenario", key=scenario["title"], use_container_width=True):
                    st.session_state.quick_query = scenario["query"]
                    st.rerun()
        
        # Technical details
        st.header("🔧 Technical Details")
        with st.expander("Architecture"):
            st.markdown("""
            - **Memory**: LangGraph MemorySaver with session isolation
            - **LLM**: Azure OpenAI GPT-4 with enhanced prompts  
            - **Tools**: Data loading, customer info, loan predictions
            - **Privacy**: Context cleared when switching customers
            """)
        
        with st.expander("What-if Analysis"):
            st.markdown("""
            - **Dynamic Parsing**: LLM interprets natural language scenarios
            - **Feature Modification**: Automatic data type handling
            - **ML Predictions**: Real models for accurate results
            - **Probability Changes**: Visual indicators with arrows and percentages
            """)


if __name__ == "__main__":
    main()
