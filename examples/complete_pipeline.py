"""
Complete example demonstrating model distillation and agentic workflow.

This script shows the full pipeline:
1. Create/load structured data
2. Distill a model using the data
3. Create an agentic workflow using the distilled model
4. Run queries through the agent
"""

import sys
import os
from pathlib import Path
import json
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_preparation import StructuredDataLoader, create_sample_dataset
from src.distillation.distiller import ModelDistiller, DistillationConfig
from src.agentic_workflow.agent import AgenticWorkflow, MultiAgentWorkflow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def step1_prepare_data(data_path: str = "data/sample_dataset.csv"):
    """Step 1: Prepare structured data for distillation."""
    logger.info("="*60)
    logger.info("STEP 1: Preparing Structured Data")
    logger.info("="*60)
    
    # Create sample dataset if it doesn't exist
    if not Path(data_path).exists():
        logger.info("Creating sample dataset...")
        os.makedirs(Path(data_path).parent, exist_ok=True)
        create_sample_dataset(data_path, num_samples=50)
    
    # Load and prepare data
    loader = StructuredDataLoader()
    df = loader.load_from_csv(data_path)
    
    logger.info(f"\nDataset preview:")
    logger.info(f"\n{df.head()}")
    
    # Create QA pairs
    qa_pairs = loader.create_qa_pairs(
        question_col="question",
        answer_col="answer",
        context_col="context"
    )
    
    # Prepare distillation dataset
    distillation_data = loader.prepare_distillation_dataset(
        text_col="question"
    )
    
    logger.info(f"\nPrepared {len(distillation_data)} examples for distillation")
    logger.info(f"Example: {distillation_data[0]}")
    
    return distillation_data, qa_pairs


def step2_distill_model(
    training_data,
    config_path: str = "config/distillation_config.json"
):
    """Step 2: Distill the model using structured data."""
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Distilling Model")
    logger.info("="*60)
    
    # Load configuration
    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = DistillationConfig(**config_dict)
    else:
        # Use default configuration
        config = DistillationConfig(
            teacher_model_name="gpt2",
            student_model_name="distilgpt2",
            temperature=2.0,
            alpha=0.5,
            max_length=128,
            batch_size=2,  # Small batch size for demo
            num_epochs=1,  # Few epochs for demo
            learning_rate=5e-5,
            output_dir="./models/distilled_model"
        )
    
    logger.info(f"\nDistillation Configuration:")
    logger.info(f"  Teacher: {config.teacher_model_name}")
    logger.info(f"  Student: {config.student_model_name}")
    logger.info(f"  Temperature: {config.temperature}")
    logger.info(f"  Alpha: {config.alpha}")
    logger.info(f"  Epochs: {config.num_epochs}")
    
    # Initialize distiller
    distiller = ModelDistiller(config)
    
    # Load models
    distiller.load_teacher_model()
    distiller.load_student_model()
    
    # Perform distillation
    logger.info("\nStarting distillation process...")
    distiller.distill(training_data[:10])  # Use subset for demo
    
    # Save distilled model
    output_dir = distiller.save_distilled_model()
    logger.info(f"\nDistilled model saved to: {output_dir}")
    
    return str(output_dir)


def step3_create_agentic_workflow(model_path: str):
    """Step 3: Create agentic workflow using the distilled model."""
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Creating Agentic Workflow")
    logger.info("="*60)
    
    # Initialize the agentic workflow
    logger.info(f"Loading distilled model from: {model_path}")
    workflow = AgenticWorkflow(model_path)
    
    logger.info("Agentic workflow created successfully!")
    return workflow


def step4_run_queries(workflow: AgenticWorkflow, qa_pairs):
    """Step 4: Run queries through the agentic workflow."""
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Running Queries Through Agent")
    logger.info("="*60)
    
    # Sample queries
    test_queries = [
        "What is artificial intelligence?",
        "Explain machine learning in simple terms.",
        "What is photosynthesis?",
    ]
    
    # Also use some from QA pairs if available
    if qa_pairs:
        test_queries.extend([qa["question"] for qa in qa_pairs[:2]])
    
    results = []
    for i, query in enumerate(test_queries[:5], 1):  # Limit to 5 queries
        logger.info(f"\n--- Query {i} ---")
        logger.info(f"Question: {query}")
        
        # Run through workflow
        result = workflow.run(query)
        
        logger.info(f"Response: {result['response']}")
        results.append(result)
    
    return results


def step5_demonstrate_multi_agent(model_path: str):
    """Step 5: Demonstrate multi-agent workflow."""
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Multi-Agent Workflow Demo")
    logger.info("="*60)
    
    # Initialize multi-agent workflow
    multi_workflow = MultiAgentWorkflow(model_path)
    
    # Test different types of queries
    queries = [
        "What is machine learning?",
        "Summarize: Machine learning is a subset of AI that enables systems to learn from data.",
        "Validate this statement: Deep learning uses neural networks.",
    ]
    
    for i, query in enumerate(queries, 1):
        logger.info(f"\n--- Multi-Agent Query {i} ---")
        logger.info(f"Query: {query}")
        
        result = multi_workflow.run(query)
        
        logger.info(f"Task Type: {result['task_type']}")
        logger.info(f"Response: {result['response']}")


def main():
    """Run the complete pipeline."""
    logger.info("="*60)
    logger.info("MODEL DISTILLATION WITH AGENTIC WORKFLOW")
    logger.info("Complete Pipeline Demo")
    logger.info("="*60)
    
    try:
        # Step 1: Prepare data
        training_data, qa_pairs = step1_prepare_data()
        
        # Step 2: Distill model
        model_path = step2_distill_model(training_data)
        
        # Step 3: Create agentic workflow
        workflow = step3_create_agentic_workflow(model_path)
        
        # Step 4: Run queries
        results = step4_run_queries(workflow, qa_pairs)
        
        # Step 5: Multi-agent demo
        step5_demonstrate_multi_agent(model_path)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"\nProcessed {len(results)} queries")
        logger.info(f"Distilled model saved to: {model_path}")
        
    except Exception as e:
        logger.error(f"\nError in pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
