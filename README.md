# Model Distillation with Agentic Workflow

A comprehensive framework for distilling large language models using structured data and deploying them through intelligent agentic workflows powered by LangGraph.

## Overview

This project demonstrates the complete pipeline for:
1. **Model Distillation**: Transfer knowledge from larger "teacher" models to smaller "student" models
2. **Structured Data Processing**: Prepare and utilize structured datasets for effective distillation
3. **Agentic Workflows**: Deploy distilled models through LangGraph-based agent workflows for intelligent task execution

## Features

- 🎓 **Knowledge Distillation**: Efficient transfer of knowledge from teacher to student models
- 📊 **Structured Data Support**: CSV, JSON, and Parquet data formats
- 🤖 **LangGraph Integration**: Build sophisticated agent workflows
- 🔄 **Multi-Agent Workflows**: Support for complex multi-step reasoning
- ⚙️ **Configurable Pipeline**: Easy configuration through JSON files
- 📈 **Training Monitoring**: Built-in logging and progress tracking

## Architecture

```
┌─────────────────┐
│ Structured Data │
└────────┬────────┘
         │
         ▼
┌─────────────────────────┐
│  Data Preparation       │
│  (CSV/JSON → Datasets)  │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Model Distillation     │
│  (Teacher → Student)    │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Distilled Model        │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  LangGraph Workflow     │
│  (Agentic Processing)   │
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Intelligent Responses  │
└─────────────────────────┘
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/nitsourish/Model-Distillation-Agentic-Workflow.git
cd Model-Distillation-Agentic-Workflow
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the quick start example to see the framework in action:

```bash
python examples/quickstart.py
```

This will:
- Create sample structured data
- Show the distillation configuration
- Demonstrate the workflow concept

## Usage

### 1. Prepare Structured Data

```python
from src.data.data_preparation import StructuredDataLoader, create_sample_dataset

# Create sample dataset
create_sample_dataset("data/sample.csv", num_samples=100)

# Load and prepare data
loader = StructuredDataLoader()
loader.load_from_csv("data/sample.csv")
qa_pairs = loader.create_qa_pairs(
    question_col="question",
    answer_col="answer"
)
```

### 2. Distill a Model

```python
from src.distillation.distiller import ModelDistiller, DistillationConfig

# Configure distillation
config = DistillationConfig(
    teacher_model_name="gpt2",
    student_model_name="distilgpt2",
    temperature=2.0,
    alpha=0.5,
    num_epochs=3
)

# Distill the model
distiller = ModelDistiller(config)
distiller.load_teacher_model()
distiller.load_student_model()
distiller.distill(training_data)
distiller.save_distilled_model("./models/my_distilled_model")
```

### 3. Create an Agentic Workflow

```python
from src.agentic_workflow.agent import AgenticWorkflow

# Initialize workflow with distilled model
workflow = AgenticWorkflow("./models/my_distilled_model")

# Run queries
result = workflow.run(
    query="What is machine learning?",
    context="AI and technology context"
)

print(result["response"])
```

### 4. Multi-Agent Workflow

```python
from src.agentic_workflow.agent import MultiAgentWorkflow

# Initialize multi-agent workflow
workflow = MultiAgentWorkflow("./models/my_distilled_model")

# Run different types of tasks
result = workflow.run("Summarize: Machine learning enables computers to learn.")
print(f"Task: {result['task_type']}, Response: {result['response']}")
```

## Complete Pipeline

Run the complete pipeline example:

```bash
python examples/complete_pipeline.py
```

This demonstrates the entire workflow from data preparation to agent-based inference.

## Configuration

### Distillation Configuration (`config/distillation_config.json`)

```json
{
  "teacher_model_name": "gpt2",
  "student_model_name": "distilgpt2",
  "temperature": 2.0,
  "alpha": 0.5,
  "max_length": 128,
  "batch_size": 8,
  "num_epochs": 3,
  "learning_rate": 5e-5,
  "output_dir": "./models/distilled_model"
}
```

### Agent Configuration (`config/agent_config.json`)

```json
{
  "model_path": "./models/distilled_model",
  "device": "cpu",
  "max_length": 100,
  "temperature": 0.7,
  "workflow_type": "single_agent"
}
```

## Project Structure

```
Model-Distillation-Agentic-Workflow/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── data_preparation.py      # Data loading and preprocessing
│   ├── distillation/
│   │   ├── __init__.py
│   │   └── distiller.py             # Model distillation logic
│   └── agentic_workflow/
│       ├── __init__.py
│       └── agent.py                 # LangGraph agent workflows
├── examples/
│   ├── quickstart.py                # Quick start example
│   └── complete_pipeline.py         # Full pipeline demo
├── config/
│   ├── distillation_config.json     # Distillation parameters
│   └── agent_config.json            # Agent configuration
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Key Components

### Data Preparation
- **StructuredDataLoader**: Load data from CSV, JSON, or Parquet
- **create_qa_pairs**: Convert structured data to QA pairs
- **prepare_distillation_dataset**: Format data for model training

### Model Distillation
- **DistillationConfig**: Configuration for the distillation process
- **ModelDistiller**: Main distillation engine
- **DistillationLoss**: Custom loss function for knowledge transfer

### Agentic Workflow
- **AgenticWorkflow**: Single-agent LangGraph workflow
- **MultiAgentWorkflow**: Multi-agent workflow with routing
- **DistilledModelAgent**: Agent that uses distilled models for inference

## How It Works

### Model Distillation

Knowledge distillation works by:
1. **Soft Targets**: The teacher model produces probability distributions over tokens
2. **Temperature Scaling**: Soften the distributions using temperature parameter
3. **Loss Combination**: Combine distillation loss with standard training loss
4. **Knowledge Transfer**: Student learns to mimic teacher's behavior

Formula:
```
L_total = α * L_distill + (1-α) * L_student
L_distill = KL_divergence(student_logits/T, teacher_logits/T) * T²
```

### Agentic Workflow

The LangGraph workflow:
1. **Process Input**: Parse and understand user query
2. **Route Task**: Determine the appropriate agent (QA, summarization, validation)
3. **Generate Response**: Use distilled model for inference
4. **Validate Output**: Check and finalize the response

## Benefits

- 🚀 **Efficiency**: Smaller models with comparable performance
- 💰 **Cost-Effective**: Reduced inference costs
- ⚡ **Fast**: Lower latency for real-time applications
- 🎯 **Focused**: Models trained on specific structured data
- 🔧 **Flexible**: Easy to customize and extend

## Examples and Use Cases

- **Question Answering**: Build efficient QA systems from large knowledge bases
- **Text Summarization**: Create compact summarization models
- **Classification**: Distill classifiers for specific domains
- **Multi-task Learning**: Single distilled model for multiple tasks

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

Sourish Dey

## Acknowledgments

- HuggingFace Transformers for model implementations
- LangChain and LangGraph for agent frameworks
- PyTorch for deep learning capabilities
