"""
Data preparation utilities for model distillation.
Handles structured data loading, preprocessing, and dataset creation.
"""

import pandas as pd
import json
from typing import List, Dict, Optional, Union
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructuredDataLoader:
    """Load and prepare structured data for model distillation."""
    
    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to structured data file (CSV, JSON, or Parquet)
        """
        self.data_path = Path(data_path) if data_path else None
        self.data = None
        
    def load_from_csv(self, file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional arguments for pd.read_csv
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from CSV: {file_path}")
        self.data = pd.read_csv(file_path, **kwargs)
        logger.info(f"Loaded {len(self.data)} records")
        return self.data
    
    def load_from_json(self, file_path: Union[str, Path]) -> pd.DataFrame:
        """
        Load data from JSON file.
        
        Args:
            file_path: Path to JSON file
            
        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading data from JSON: {file_path}")
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.data = pd.DataFrame(data)
        elif isinstance(data, dict):
            self.data = pd.DataFrame([data])
        
        logger.info(f"Loaded {len(self.data)} records")
        return self.data
    
    def create_qa_pairs(
        self,
        question_col: str = "question",
        answer_col: str = "answer",
        context_col: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Create question-answer pairs from structured data.
        
        Args:
            question_col: Column name for questions
            answer_col: Column name for answers
            context_col: Optional column name for context
            
        Returns:
            List of QA pair dictionaries
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_from_csv or load_from_json first.")
        
        qa_pairs = []
        for _, row in self.data.iterrows():
            pair = {
                "question": str(row[question_col]),
                "answer": str(row[answer_col])
            }
            if context_col and context_col in self.data.columns:
                pair["context"] = str(row[context_col])
            qa_pairs.append(pair)
        
        logger.info(f"Created {len(qa_pairs)} QA pairs")
        return qa_pairs
    
    def prepare_distillation_dataset(
        self,
        text_col: str = "text",
        label_col: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Prepare dataset for distillation training.
        
        Args:
            text_col: Column name for text data
            label_col: Optional column name for labels
            
        Returns:
            List of training examples
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_from_csv or load_from_json first.")
        
        dataset = []
        for _, row in self.data.iterrows():
            example = {"text": str(row[text_col])}
            if label_col and label_col in self.data.columns:
                example["label"] = str(row[label_col])
            dataset.append(example)
        
        logger.info(f"Prepared {len(dataset)} training examples")
        return dataset


def create_sample_dataset(output_path: Union[str, Path], num_samples: int = 100):
    """
    Create a sample structured dataset for demonstration.
    
    Args:
        output_path: Path to save the sample dataset
        num_samples: Number of samples to generate
    """
    import random
    
    # Sample topics and templates
    topics = ["technology", "science", "history", "geography", "literature"]
    
    questions = []
    answers = []
    contexts = []
    
    templates = {
        "technology": [
            ("What is artificial intelligence?", 
             "Artificial intelligence is the simulation of human intelligence by machines."),
            ("Explain machine learning.", 
             "Machine learning is a subset of AI that enables systems to learn from data."),
            ("What is deep learning?",
             "Deep learning uses neural networks with multiple layers to learn patterns."),
        ],
        "science": [
            ("What is photosynthesis?",
             "Photosynthesis is the process by which plants convert light into energy."),
            ("Explain the water cycle.",
             "The water cycle describes how water moves between Earth's oceans, atmosphere, and land."),
            ("What is gravity?",
             "Gravity is a force that attracts objects with mass toward each other."),
        ],
        "history": [
            ("When was the Declaration of Independence signed?",
             "The Declaration of Independence was signed on July 4, 1776."),
            ("Who was the first President of the United States?",
             "George Washington was the first President of the United States."),
            ("What caused World War I?",
             "World War I was caused by a complex mix of alliances, militarism, and nationalism."),
        ],
    }
    
    for _ in range(num_samples):
        topic = random.choice(topics)
        if topic in templates:
            q, a = random.choice(templates[topic])
            questions.append(q)
            answers.append(a)
            contexts.append(f"This is a question about {topic}.")
    
    df = pd.DataFrame({
        "question": questions,
        "answer": answers,
        "context": contexts,
        "topic": [random.choice(topics) for _ in range(len(questions))]
    })
    
    df.to_csv(output_path, index=False)
    logger.info(f"Created sample dataset with {len(df)} records at {output_path}")
    return df
