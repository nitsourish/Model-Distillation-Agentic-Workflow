"""
Model distillation implementation for knowledge transfer from teacher to student models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from typing import Dict, List, Optional, Union
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DistillationConfig:
    """Configuration for model distillation."""
    
    def __init__(
        self,
        teacher_model_name: str = "gpt2",
        student_model_name: str = "distilgpt2",
        temperature: float = 2.0,
        alpha: float = 0.5,
        max_length: int = 512,
        batch_size: int = 8,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,
        output_dir: str = "./distilled_model",
    ):
        """
        Initialize distillation configuration.
        
        Args:
            teacher_model_name: Name or path of the teacher model
            student_model_name: Name or path of the student model
            temperature: Temperature for distillation (higher = softer probabilities)
            alpha: Weight for distillation loss (1-alpha for student loss)
            max_length: Maximum sequence length
            batch_size: Training batch size
            num_epochs: Number of training epochs
            learning_rate: Learning rate for training
            output_dir: Directory to save the distilled model
        """
        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.temperature = temperature
        self.alpha = alpha
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.output_dir = output_dir
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary."""
        return {
            "teacher_model_name": self.teacher_model_name,
            "student_model_name": self.student_model_name,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "max_length": self.max_length,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "learning_rate": self.learning_rate,
            "output_dir": self.output_dir,
        }
    
    def save(self, path: Union[str, Path]):
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)


class DistillationLoss(nn.Module):
    """Custom loss for knowledge distillation."""
    
    def __init__(self, temperature: float = 2.0, alpha: float = 0.5):
        """
        Initialize distillation loss.
        
        Args:
            temperature: Temperature for softening probability distributions
            alpha: Weight for distillation loss (1-alpha for hard label loss)
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Logits from student model
            teacher_logits: Logits from teacher model
            labels: Ground truth labels (optional)
            
        Returns:
            Combined distillation loss
        """
        # Soft targets loss (distillation loss)
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_prob = F.log_softmax(student_logits / self.temperature, dim=-1)
        distillation_loss = F.kl_div(
            soft_prob,
            soft_targets,
            reduction='batchmean'
        ) * (self.temperature ** 2)
        
        # Hard targets loss (student loss)
        if labels is not None:
            student_loss = F.cross_entropy(student_logits, labels)
            total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        else:
            total_loss = distillation_loss
        
        return total_loss


class ModelDistiller:
    """Main class for model distillation."""
    
    def __init__(self, config: DistillationConfig):
        """
        Initialize the model distiller.
        
        Args:
            config: Distillation configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize models
        self.teacher_model = None
        self.student_model = None
        self.loss_fn = DistillationLoss(
            temperature=config.temperature,
            alpha=config.alpha
        )
    
    def load_teacher_model(self):
        """Load the teacher model."""
        logger.info(f"Loading teacher model: {self.config.teacher_model_name}")
        self.teacher_model = AutoModelForCausalLM.from_pretrained(
            self.config.teacher_model_name
        )
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        logger.info("Teacher model loaded successfully")
    
    def load_student_model(self):
        """Load the student model."""
        logger.info(f"Loading student model: {self.config.student_model_name}")
        self.student_model = AutoModelForCausalLM.from_pretrained(
            self.config.student_model_name
        )
        self.student_model.to(self.device)
        self.student_model.train()
        logger.info("Student model loaded successfully")
    
    def prepare_data(self, texts: List[str]) -> Dict:
        """
        Tokenize input texts.
        
        Args:
            texts: List of input texts
            
        Returns:
            Tokenized inputs
        """
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt"
        )
    
    def distill(self, training_data: List[Dict[str, str]]):
        """
        Perform model distillation.
        
        Args:
            training_data: List of training examples with 'text' field
        """
        if self.teacher_model is None:
            self.load_teacher_model()
        if self.student_model is None:
            self.load_student_model()
        
        logger.info(f"Starting distillation with {len(training_data)} examples")
        
        # Extract texts from training data
        texts = [example.get("text", "") for example in training_data]
        
        # Simple distillation loop (for demonstration)
        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config.learning_rate
        )
        
        for epoch in range(self.config.num_epochs):
            total_loss = 0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(texts), self.config.batch_size):
                batch_texts = texts[i:i + self.config.batch_size]
                
                # Tokenize
                inputs = self.prepare_data(batch_texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Get teacher outputs (no gradients)
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(**inputs)
                    teacher_logits = teacher_outputs.logits
                
                # Get student outputs
                student_outputs = self.student_model(**inputs)
                student_logits = student_outputs.logits
                
                # Compute loss (only on last token for simplicity)
                loss = self.loss_fn(
                    student_logits[:, -1, :],
                    teacher_logits[:, -1, :]
                )
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}, Loss: {avg_loss:.4f}")
        
        logger.info("Distillation completed")
    
    def save_distilled_model(self, output_dir: Optional[str] = None):
        """
        Save the distilled student model.
        
        Args:
            output_dir: Directory to save the model (uses config.output_dir if None)
        """
        save_dir = Path(output_dir or self.config.output_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving distilled model to {save_dir}")
        self.student_model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # Save configuration
        config_path = save_dir / "distillation_config.json"
        self.config.save(config_path)
        
        logger.info("Model saved successfully")
        return save_dir
    
    def load_distilled_model(self, model_path: Union[str, Path]):
        """
        Load a previously distilled model.
        
        Args:
            model_path: Path to the distilled model
        """
        logger.info(f"Loading distilled model from {model_path}")
        self.student_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.student_model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logger.info("Distilled model loaded successfully")


def create_distiller_from_config(config_dict: Dict) -> ModelDistiller:
    """
    Create a ModelDistiller from a configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        ModelDistiller instance
    """
    config = DistillationConfig(**config_dict)
    return ModelDistiller(config)
