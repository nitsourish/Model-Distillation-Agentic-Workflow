"""
Test script to verify the implementation without heavy ML dependencies.

This script tests the core functionality:
- Data loading and preparation
- Configuration handling
- Module structure

For full testing including model distillation and agents, install all dependencies:
    pip install -r requirements.txt
"""

import sys
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_preparation import StructuredDataLoader, create_sample_dataset

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_preparation():
    """Test data preparation functionality."""
    logger.info("="*60)
    logger.info("TEST 1: Data Preparation")
    logger.info("="*60)
    
    try:
        # Create sample dataset
        Path("data").mkdir(exist_ok=True)
        df = create_sample_dataset("data/test_dataset.csv", num_samples=20)
        assert len(df) > 0, "Dataset should not be empty"
        logger.info(f"✓ Created dataset with {len(df)} records")
        
        # Load dataset
        loader = StructuredDataLoader()
        loaded_df = loader.load_from_csv("data/test_dataset.csv")
        assert len(loaded_df) == len(df), "Loaded data should match created data"
        logger.info(f"✓ Loaded dataset successfully")
        
        # Create QA pairs
        qa_pairs = loader.create_qa_pairs(
            question_col="question",
            answer_col="answer",
            context_col="context"
        )
        assert len(qa_pairs) > 0, "Should create QA pairs"
        assert "question" in qa_pairs[0], "QA pair should have question"
        assert "answer" in qa_pairs[0], "QA pair should have answer"
        logger.info(f"✓ Created {len(qa_pairs)} QA pairs")
        
        # Prepare distillation dataset
        training_data = loader.prepare_distillation_dataset(text_col="question")
        assert len(training_data) == len(df), "Training data should match input"
        assert "text" in training_data[0], "Training data should have text field"
        logger.info(f"✓ Prepared {len(training_data)} training examples")
        
        logger.info("\n✅ Data preparation tests PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Data preparation tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_configuration():
    """Test configuration loading."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Configuration")
    logger.info("="*60)
    
    try:
        # Test distillation config
        distill_config_path = "config/distillation_config.json"
        with open(distill_config_path, 'r') as f:
            distill_config = json.load(f)
        
        required_keys = [
            "teacher_model_name", "student_model_name", "temperature",
            "alpha", "num_epochs", "output_dir"
        ]
        for key in required_keys:
            assert key in distill_config, f"Config should have {key}"
        
        logger.info(f"✓ Distillation config valid")
        logger.info(f"  Teacher: {distill_config['teacher_model_name']}")
        logger.info(f"  Student: {distill_config['student_model_name']}")
        
        # Test agent config
        agent_config_path = "config/agent_config.json"
        with open(agent_config_path, 'r') as f:
            agent_config = json.load(f)
        
        assert "model_path" in agent_config, "Agent config should have model_path"
        logger.info(f"✓ Agent config valid")
        logger.info(f"  Model path: {agent_config['model_path']}")
        
        logger.info("\n✅ Configuration tests PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Configuration tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_module_structure():
    """Test that all modules can be imported."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Module Structure")
    logger.info("="*60)
    
    try:
        # Test data module
        from src.data.data_preparation import (
            StructuredDataLoader,
            create_sample_dataset
        )
        logger.info("✓ Data module imports successfully")
        
        # Test distillation module (may fail if torch not installed)
        try:
            from src.distillation.distiller import (
                ModelDistiller,
                DistillationConfig,
                DistillationLoss
            )
            logger.info("✓ Distillation module imports successfully")
        except ImportError as e:
            logger.warning(f"⚠ Distillation module requires torch: {e}")
        
        # Test agent module (may fail if langgraph not installed)
        try:
            from src.agentic_workflow.agent import (
                AgenticWorkflow,
                MultiAgentWorkflow,
                DistilledModelAgent
            )
            logger.info("✓ Agent workflow module imports successfully")
        except ImportError as e:
            logger.warning(f"⚠ Agent module requires langgraph: {e}")
        
        logger.info("\n✅ Module structure tests PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Module structure tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_example_scripts():
    """Test that example scripts exist and are valid."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Example Scripts")
    logger.info("="*60)
    
    try:
        examples = [
            "examples/quickstart.py",
            "examples/complete_pipeline.py",
            "examples/custom_data_example.py"
        ]
        
        for example in examples:
            path = Path(example)
            assert path.exists(), f"Example {example} should exist"
            # Check it's valid Python
            with open(path, 'r') as f:
                content = f.read()
                compile(content, path.name, 'exec')
            logger.info(f"✓ {example} exists and is valid Python")
        
        logger.info("\n✅ Example scripts tests PASSED")
        return True
        
    except Exception as e:
        logger.error(f"\n❌ Example scripts tests FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    logger.info("\n" + "="*60)
    logger.info("MODEL DISTILLATION FRAMEWORK - TEST SUITE")
    logger.info("="*60)
    
    results = {
        "Data Preparation": test_data_preparation(),
        "Configuration": test_configuration(),
        "Module Structure": test_module_structure(),
        "Example Scripts": test_example_scripts(),
    }
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST SUMMARY")
    logger.info("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        logger.info("\n🎉 All tests PASSED!")
        logger.info("\nTo test full functionality including model distillation and agents:")
        logger.info("  pip install -r requirements.txt")
        logger.info("  python examples/complete_pipeline.py")
        return 0
    else:
        logger.error("\n❌ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
