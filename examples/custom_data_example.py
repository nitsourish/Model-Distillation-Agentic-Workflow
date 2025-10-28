"""
Example: Using your own structured data for distillation.

This example shows how to:
1. Prepare your own CSV/JSON data
2. Load and process it
3. Distill a model with your data
4. Use the distilled model in an agent
"""

import sys
from pathlib import Path
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_preparation import StructuredDataLoader
from src.distillation.distiller import ModelDistiller, DistillationConfig
from src.agentic_workflow.agent import AgenticWorkflow

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_custom_dataset():
    """Create a custom dataset example."""
    # Example: Create a FAQ dataset for a specific domain
    data = {
        "question": [
            "What are the business hours?",
            "How do I reset my password?",
            "What payment methods do you accept?",
            "How can I track my order?",
            "What is your return policy?",
            "Do you offer international shipping?",
            "How do I contact customer support?",
            "What is the warranty period?",
            "Can I modify my order after placing it?",
            "How long does shipping take?",
        ],
        "answer": [
            "Our business hours are Monday to Friday, 9 AM to 6 PM EST.",
            "Click on 'Forgot Password' on the login page and follow the instructions.",
            "We accept credit cards, PayPal, and Apple Pay.",
            "You can track your order using the tracking link sent to your email.",
            "We accept returns within 30 days of purchase for a full refund.",
            "Yes, we ship to over 50 countries worldwide.",
            "You can reach us at support@example.com or call 1-800-EXAMPLE.",
            "All products come with a 1-year manufacturer warranty.",
            "Contact us within 1 hour of placing the order to modify it.",
            "Standard shipping takes 5-7 business days.",
        ],
        "category": [
            "general", "account", "payment", "orders", "returns",
            "shipping", "support", "warranty", "orders", "shipping"
        ],
    }
    
    df = pd.DataFrame(data)
    output_path = "data/custom_faq_dataset.csv"
    Path(output_path).parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    
    logger.info(f"Created custom dataset with {len(df)} FAQ entries")
    logger.info(f"Saved to: {output_path}")
    
    return output_path


def main():
    """Main workflow with custom data."""
    logger.info("="*60)
    logger.info("Using Custom Data for Model Distillation")
    logger.info("="*60)
    
    # 1. Create or load custom data
    logger.info("\n1. Creating custom FAQ dataset...")
    data_path = create_custom_dataset()
    
    # 2. Load and prepare the data
    logger.info("\n2. Loading and preparing data...")
    loader = StructuredDataLoader()
    df = loader.load_from_csv(data_path)
    
    logger.info(f"\nDataset info:")
    logger.info(f"  Total records: {len(df)}")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"\nSample records:")
    logger.info(df.head(3).to_string())
    
    # Create training data
    training_data = loader.prepare_distillation_dataset(text_col="question")
    qa_pairs = loader.create_qa_pairs()
    
    logger.info(f"\nPrepared {len(training_data)} training examples")
    
    # 3. Configure and run distillation (conceptual)
    logger.info("\n3. Distillation Configuration:")
    config = DistillationConfig(
        teacher_model_name="gpt2",
        student_model_name="distilgpt2",
        temperature=2.0,
        alpha=0.5,
        max_length=128,
        batch_size=4,
        num_epochs=3,
        output_dir="./models/custom_faq_model"
    )
    
    logger.info(f"  Teacher: {config.teacher_model_name}")
    logger.info(f"  Student: {config.student_model_name}")
    logger.info(f"  Training examples: {len(training_data)}")
    
    logger.info("\n  To actually perform distillation, uncomment the following:")
    logger.info("  # distiller = ModelDistiller(config)")
    logger.info("  # distiller.load_teacher_model()")
    logger.info("  # distiller.load_student_model()")
    logger.info("  # distiller.distill(training_data)")
    logger.info("  # model_path = distiller.save_distilled_model()")
    
    # 4. Demonstrate how to use the distilled model
    logger.info("\n4. Using the Distilled Model (Conceptual):")
    logger.info("  Once distilled, you would use it like this:")
    logger.info("  # workflow = AgenticWorkflow(model_path)")
    logger.info("  # result = workflow.run('What are your business hours?')")
    logger.info("  # print(result['response'])")
    
    # Show example queries that would be answered
    logger.info("\n5. Example queries for your distilled FAQ bot:")
    example_queries = [
        "What are the business hours?",
        "How do I reset my password?",
        "What payment methods are available?",
    ]
    
    for i, query in enumerate(example_queries, 1):
        logger.info(f"  {i}. {query}")
        # Find matching answer from QA pairs
        matching = [qa for qa in qa_pairs if qa['question'] == query]
        if matching:
            logger.info(f"     Expected: {matching[0]['answer'][:60]}...")
    
    logger.info("\n" + "="*60)
    logger.info("Custom Data Workflow Complete!")
    logger.info("\nKey takeaways:")
    logger.info("- You can use any CSV/JSON data with question-answer pairs")
    logger.info("- The data_preparation module handles loading and formatting")
    logger.info("- Distillation creates a specialized model for your domain")
    logger.info("- The agent workflow makes it easy to deploy and use")
    logger.info("="*60)


if __name__ == "__main__":
    main()
