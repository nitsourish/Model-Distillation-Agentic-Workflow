"""
Tool definitions for loan prediction application.
"""

import json
import pandas as pd
import re
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

from utils import _make_json_serializable, format_probability_change
from prediction_utils import (
    get_prediction_service
)
from prediction_utils import predict_customer_loan_amount

# Global variable for data storage
_loaded_data = None


@tool
def load_data() -> str:
    """Load the loan dataset from a CSV file."""
    import os
    global _loaded_data
    
    # Construct relative path from the project root
    current_dir = os.path.dirname(os.path.abspath(__file__))  # scripts directory
    project_root = os.path.dirname(current_dir)  # model_distillation directory
    data_path = os.path.join(project_root, "data", "business_simulation.csv")
    
    _loaded_data = pd.read_csv(data_path)
    return f"Successfully loaded dataset with {len(_loaded_data)} customers"


@tool
def get_customer_information(customer_id: int) -> str:
    """Get customer information including current loan status."""
    global _loaded_data
    if '_loaded_data' not in globals() or _loaded_data is None:
        return "Error: Data not loaded. Please call load_data first."
    
    customer_info = _loaded_data[_loaded_data['customer_id'] == customer_id]
    if customer_info.empty:
        return f"Error: Customer ID {customer_id} not found in the dataset. Please provide a valid customer ID."
    
    return customer_info.to_json(orient='records', indent=2)


@tool
def load_predict_loan_approval_propensity(customer_id: int) -> str:
    """Fast loan approval propensity prediction."""
    global _loaded_data
    if '_loaded_data' not in globals() or _loaded_data is None:
        return "Error: Data not loaded. Please call load_data first."
    
    # Initialize prediction service if needed
    prediction_service = get_prediction_service()
    
    # Make prediction
    result = predict_customer_loan_propensity(_loaded_data, customer_id, prediction_service)
    
    if "error" in result:
        return str(result)
    
    # Convert to JSON-serializable format
    result = _make_json_serializable(result)
    
    return json.dumps(result, indent=2)


@tool   
def load_predict_loan_amount(customer_id: int) -> str:
    """Predict loan amount for a specific customer using ML model."""
    global _loaded_data
    if '_loaded_data' not in globals() or _loaded_data is None:
        return "Error: Data not loaded. Please call load_data first."
    
    # Initialize prediction service if needed
    prediction_service = get_prediction_service()
    
    # Make prediction using the proper ML-based function
    from prediction_utils import predict_customer_loan_amount
    result = predict_customer_loan_amount(_loaded_data, customer_id, prediction_service)
    
    if "error" in result:
        return str(result)
    
    # Convert to JSON-serializable format
    result = _make_json_serializable(result)
    return json.dumps(result, indent=2)


def create_whatif_tools(model):
    """Create whatif tools with model dependency."""
    
    @tool
    def load_predict_loan_approval_whatif(customer_id: int, scenario_description: str = "") -> str:
        """
        Predict loan approval with intelligent natural language scenario modifications.
        Uses LLM to dynamically modify customer data based on scenario description.
        """
        global _loaded_data
        if '_loaded_data' not in globals() or _loaded_data is None:
            return "Error: Data not loaded. Please call load_data first."
        
        # Get customer data
        customer_data = _loaded_data.loc[_loaded_data['customer_id'] == customer_id]
        if customer_data.empty:
            return f"Error: Customer ID {customer_id} not found in the dataset. Please provide a valid customer ID."
        
        modified_data = customer_data.copy()
        modifications_applied = []
        
        # If no scenario, return baseline
        if not scenario_description or not scenario_description.strip():
            prediction_service = get_prediction_service()
            result = prediction_service.predict_single_customer(customer_data)
            if "error" in result:
                return str(result)
            
            result = _make_json_serializable(result)
            result['customer_id'] = customer_id
            result['scenario_description'] = "baseline"
            result['modifications_applied'] = []
            return json.dumps(result, indent=2)
        
        # LLM-powered data modification
        if model:
            try:
                # Get dataset structure
                column_info = {}
                for col in _loaded_data.columns:
                    if col not in ['customer_id', 'loan_status', 'max_loan']:
                        if _loaded_data[col].dtype == 'object':
                            unique_vals = list(_loaded_data[col].unique())
                            column_info[col] = {'type': 'categorical', 'values': unique_vals[:10]}
                        else:
                            column_info[col] = {
                                'type': 'numerical',
                                'min': float(_loaded_data[col].min()),
                                'max': float(_loaded_data[col].max()),
                                'current': float(customer_data[col].iloc[0])
                            }
                
                # Create LLM prompt for parsing
                modification_prompt = f"""
Analyze this scenario and determine feature modifications for a loan customer.

CURRENT DATA: {customer_data.iloc[0].to_dict()}
AVAILABLE FEATURES: {json.dumps(column_info, indent=2)}
SCENARIO: "{scenario_description}"

Return JSON with modifications:
{{
    "modifications": {{
        "column_name": "new_value"
    }}
}}

Rules:
- Use exact column names from available features
- For categorical: use values from the available list
- For numerical: use reasonable values within range
- Only modify explicitly mentioned features
"""
                
                # Parse with LLM
                llm_response = model.invoke([HumanMessage(content=modification_prompt)])
                llm_content = llm_response.content.strip()
                
                json_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
                if json_match:
                    modification_data = json.loads(json_match.group())
                    
                    if 'modifications' in modification_data:
                        for column, new_value in modification_data['modifications'].items():
                            if column in modified_data.columns:
                                # Fix: update only the single row for the customer
                                col_idx = modified_data.columns.get_loc(column)
                                modified_data.iloc[0, col_idx] = new_value
                                modifications_applied.append(f"{column}={new_value}")
                                if len(modifications_applied) > 10:
                                    break
                
            except Exception:
                pass  # Use baseline data if parsing fails
        
        # Make predictions
        prediction_service = get_prediction_service()
        baseline_result = prediction_service.predict_single_customer(customer_data)
        result = prediction_service.predict_single_customer(modified_data)
        
        if "error" in result:
            return str(result)
        
        # Format results
        result = _make_json_serializable(result)
        result['customer_id'] = customer_id
        result['scenario_description'] = scenario_description
        result['modifications_applied'] = modifications_applied
        result['baseline_loan_approval_probability'] = baseline_result['loan_approval_probability']
        result['new_loan_approval_probability'] = result['loan_approval_probability']
        result['probability_change'] = format_probability_change(
            result['loan_approval_probability'], 
            baseline_result['loan_approval_probability']
        )
        
        return json.dumps(result, indent=2)

    @tool   
    def load_predict_loan_amount_whatif(customer_id: int, scenario_description: str = "") -> str:
        """
        Predict loan amount with intelligent natural language scenario modifications.
        Uses LLM to dynamically modify customer data based on scenario description.
        """
        global _loaded_data
        if '_loaded_data' not in globals() or _loaded_data is None:
            return "Error: Data not loaded. Please call load_data first."
        
        customer_data = _loaded_data.loc[_loaded_data['customer_id'] == customer_id]
        if customer_data.empty:
            return f"Error: Customer ID {customer_id} not found in the dataset. Please provide a valid customer ID."
        
        modified_data = customer_data.copy()
        modifications_applied = []
        
        # Apply same LLM parsing logic as approval function
        if scenario_description and scenario_description.strip() and model:
            try:
                column_info = {}
                for col in _loaded_data.columns:
                    if col not in ['customer_id', 'loan_status', 'max_loan']:
                        if _loaded_data[col].dtype == 'object':
                            unique_vals = list(_loaded_data[col].unique())
                            column_info[col] = {'type': 'categorical', 'values': unique_vals[:10]}
                        else:
                            column_info[col] = {
                                'type': 'numerical',
                                'min': float(_loaded_data[col].min()),
                                'max': float(_loaded_data[col].max()),
                                'current': float(customer_data[col].iloc[0])
                            }
                
                modification_prompt = f"""
You are analyzing a loan scenario modification request. The user is asking about a loan amount prediction.

CURRENT CUSTOMER DATA: {customer_data.iloc[0].to_dict()}
AVAILABLE FEATURES: {json.dumps(column_info, indent=2)}
USER REQUEST: "{scenario_description}"

The user might be asking about:
1. A specific new scenario with explicit changes (e.g., "what if they had Masters degree")
2. A reference to a previously discussed scenario (e.g., "this new scenario", "in this case")
3. Maximum loan amount with current/baseline data (if no specific changes mentioned)

If the request mentions "this scenario", "this new scenario", "in this case", or similar contextual references WITHOUT specifying actual changes, 
treat it as asking for the maximum loan amount with the CURRENT customer data (no modifications).

If specific changes are mentioned (education, income, job, etc.), apply those modifications.

Return JSON with modifications (empty if no specific changes mentioned):
{{
    "modifications": {{
        "column_name": "new_value"
    }},
    "scenario_type": "specific_changes" or "current_baseline"
}}
"""
                
                llm_response = model.invoke([HumanMessage(content=modification_prompt)])
                llm_content = llm_response.content.strip()
                
                json_match = re.search(r'\{.*\}', llm_content, re.DOTALL)
                if json_match:
                    modification_data = json.loads(json_match.group())
                    
                    # Check if this is asking about current baseline without specific changes
                    scenario_type = modification_data.get('scenario_type', 'specific_changes')
                    
                    if scenario_type == 'current_baseline' or not modification_data.get('modifications'):
                        # User is asking about current data without modifications
                        modifications_applied.append("Using current customer data (no modifications)")
                    else:
                        # Apply specific modifications
                        if 'modifications' in modification_data:
                            for column, new_value in modification_data['modifications'].items():
                                if column in modified_data.columns:
                                    if _loaded_data[column].dtype == 'object':
                                        modified_data[column] = str(new_value)
                                    elif 'int' in str(_loaded_data[column].dtype):
                                        modified_data[column] = int(new_value)
                                    elif 'float' in str(_loaded_data[column].dtype):
                                        modified_data[column] = float(new_value)
                                    else:
                                        modified_data[column] = new_value
                                    
                                    modifications_applied.append(f"{column}={new_value}")
                                    
                                    if len(modifications_applied) > 10:
                                        break
                
            except Exception:
                pass
        
        # Make predictions using the ML models
        prediction_service = get_prediction_service()
        
        
        # Get baseline and scenario loan amounts using proper ML model
        baseline_result = predict_customer_loan_amount(_loaded_data, customer_id, prediction_service)
        
        # For scenario, create temporary modified data
        temp_data = _loaded_data.copy()
        
        # Update the specific customer's row in temp_data with modified values
        customer_mask = temp_data['customer_id'] == customer_id
        if customer_mask.any():
            # Get the modified row data
            modified_row = modified_data.loc[modified_data.index[0]]
            
            # Update each column individually to avoid the pandas assignment error
            for col in modified_data.columns:
                if col in temp_data.columns:
                    temp_data.loc[customer_mask, col] = modified_row[col]
        
        scenario_result = predict_customer_loan_amount(temp_data, customer_id, prediction_service)
        
        # Handle errors
        if "error" in baseline_result:
            return json.dumps(baseline_result, indent=2)
        if "error" in scenario_result:
            return json.dumps(scenario_result, indent=2)
        
        result = {
            'customer_id': customer_id,
            'loan_approval_probability': scenario_result['loan_approval_probability'],
            'approved': scenario_result['approved'],
            'predicted_loan_amount': scenario_result['predicted_loan_amount'],
            'baseline_loan_amount': baseline_result['predicted_loan_amount'],
            'scenario_description': scenario_description,
            'modifications_applied': modifications_applied,
            'reason': scenario_result['reason']
        }

        # Add change calculations
        result = _make_json_serializable(result)
        result['baseline_loan_approval_probability'] = baseline_result['loan_approval_probability']
        result['new_loan_approval_probability'] = result['loan_approval_probability']
        result['probability_change'] = format_probability_change(
            result['loan_approval_probability'], 
            baseline_result['loan_approval_probability']
        )
        result['loan_amount_change'] = f"+{result['predicted_loan_amount'] - result['baseline_loan_amount']:.2f}"

        return json.dumps(result, indent=2)
    
    return load_predict_loan_approval_whatif, load_predict_loan_amount_whatif


@tool
def get_max_loan_amount(customer_id: int) -> str:
    """Get the maximum approved loan amount for a customer using ML model."""
    global _loaded_data
    if '_loaded_data' not in globals() or _loaded_data is None:
        return "Error: Data not loaded. Please call load_data first."
    
    # Initialize prediction service if needed
    prediction_service = get_prediction_service()
    
    # Make prediction using the proper ML-based function
    from prediction_utils import predict_customer_loan_amount
    result = predict_customer_loan_amount(_loaded_data, customer_id, prediction_service)
    
    if "error" in result:
        return str(result)
    
    # Rename for clarity in this context
    result['max_approved_loan_amount'] = result.pop('predicted_loan_amount')
    result['reason'] = 'Maximum loan amount based on ML model prediction' if result['approved'] else 'Loan rejected due to low approval probability (< 0.5)'
    
    # Convert to JSON-serializable format
    result = _make_json_serializable(result)
    return json.dumps(result, indent=2)


def get_all_tools(model=None):
    """Return all available tools."""
    base_tools = {
        "load_data": load_data,
        "get_customer_information": get_customer_information,
        "load_predict_loan_approval_propensity": load_predict_loan_approval_propensity,
        "load_predict_loan_amount": load_predict_loan_amount,
        "get_max_loan_amount": get_max_loan_amount,
    }
    
    if model:
        whatif_approval, whatif_amount = create_whatif_tools(model)
        base_tools.update({
            "load_predict_loan_approval_whatif": whatif_approval,
            "load_predict_loan_amount_whatif": whatif_amount
        })
    
    return base_tools
