"""
Quick start example for model distillation and agentic workflow.

This is a simplified version that demonstrates the key concepts without
running the full distillation (which can be time-consuming).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_preparation import create_sample_dataset, StructuredDataLoader
from src.distillation.distiller import DistillationConfig
from src.agentic_workflow.agent import DistilledModelAgent

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Quick start demo."""
    logger.info("Model Distillation & Agentic Workflow - Quick Start")
    logger.info("=" * 60)
    
    # 1. Create sample structured data
    logger.info("\n1. Creating sample structured data...")
    data_path = "data/quickstart_data.csv"
    Path(data_path).parent.mkdir(exist_ok=True)
    df = create_sample_dataset(data_path, num_samples=20)
    logger.info(f"Created dataset with {len(df)} records")
    logger.info(f"\nSample data:\n{df.head(3)}")
    
    # 2. Load and prepare data
    logger.info("\n2. Loading and preparing data...")
    loader = StructuredDataLoader()
    loader.load_from_csv(data_path)
    qa_pairs = loader.create_qa_pairs()
    logger.info(f"Prepared {len(qa_pairs)} QA pairs")
    logger.info(f"Example QA pair:\n  Q: {qa_pairs[0]['question']}\n  A: {qa_pairs[0]['answer']}")
    
    # 3. Show distillation configuration
    logger.info("\n3. Model Distillation Configuration:")
    config = DistillationConfig(
        teacher_model_name="gpt2",
        student_model_name="distilgpt2",
        temperature=2.0,
        alpha=0.5,
        num_epochs=3,
    )
    logger.info(f"  Teacher Model: {config.teacher_model_name}")
    logger.info(f"  Student Model: {config.student_model_name}")
    logger.info(f"  Temperature: {config.temperature}")
    logger.info(f"  Alpha (distillation weight): {config.alpha}")
    
    logger.info("\nNote: To run actual distillation, use 'complete_pipeline.py'")
    logger.info("This would:")
    logger.info("  - Load teacher and student models")
    logger.info("  - Transfer knowledge from teacher to student")
    logger.info("  - Save the distilled model")
    
    # 4. Demonstrate how the agent would work (conceptual)
    logger.info("\n4. Agentic Workflow (Conceptual):")
    logger.info("After distillation, the agent workflow would:")
    logger.info("  - Load the distilled model")
    logger.info("  - Create a LangGraph agent workflow")
    logger.info("  - Process queries through the workflow")
    logger.info("  - Generate responses using the distilled model")
    
    logger.info("\n" + "=" * 60)
    logger.info("Quick Start Complete!")
    logger.info("\nNext steps:")
    logger.info("1. Review the generated sample data in: data/quickstart_data.csv")
    logger.info("2. Run 'python examples/complete_pipeline.py' for full pipeline")
    logger.info("3. Customize config files in the 'config/' directory")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
