# Architecture Documentation

## Overview

This project implements a complete pipeline for model distillation using structured data and deploying the distilled models through intelligent agentic workflows powered by LangGraph.

## System Architecture

### 1. Data Layer

**Module**: `src/data/data_preparation.py`

The data layer handles loading and preprocessing of structured data from various sources.

#### Components:

- **StructuredDataLoader**: Main class for data loading
  - Supports CSV, JSON, and Parquet formats
  - Provides methods to create QA pairs from structured data
  - Prepares datasets for distillation training

```python
loader = StructuredDataLoader()
loader.load_from_csv("data/dataset.csv")
qa_pairs = loader.create_qa_pairs()
training_data = loader.prepare_distillation_dataset()
```

### 2. Distillation Layer

**Module**: `src/distillation/distiller.py`

The distillation layer implements knowledge transfer from teacher to student models.

#### Components:

- **DistillationConfig**: Configuration for the distillation process
- **DistillationLoss**: Custom loss function for knowledge distillation
- **ModelDistiller**: Main distillation engine

#### Knowledge Distillation Process:

1. **Initialize Models**:
   - Load teacher model (e.g., GPT-2)
   - Load student model (e.g., DistilGPT-2)

2. **Compute Soft Targets**:
   - Teacher generates probability distributions
   - Temperature scaling softens the distributions
   
3. **Train Student**:
   - Student learns from both soft targets and hard labels
   - Loss combines distillation loss and standard cross-entropy

4. **Save Model**:
   - Save distilled student model
   - Save tokenizer and configuration

```python
config = DistillationConfig(
    teacher_model_name="gpt2",
    student_model_name="distilgpt2",
    temperature=2.0,
    alpha=0.5
)

distiller = ModelDistiller(config)
distiller.load_teacher_model()
distiller.load_student_model()
distiller.distill(training_data)
distiller.save_distilled_model()
```

### 3. Agentic Workflow Layer

**Module**: `src/agentic_workflow/agent.py`

The workflow layer creates intelligent agents that use distilled models for inference.

#### Components:

- **DistilledModelAgent**: Base agent using distilled models
- **AgenticWorkflow**: Single-agent LangGraph workflow
- **MultiAgentWorkflow**: Multi-agent workflow with routing

#### Workflow States:

```python
class AgentState:
    messages: List[BaseMessage]
    context: str
    current_task: Optional[str]
    results: List[str]
```

#### Workflow Nodes:

1. **Process Input**: Parse user query
2. **Generate Response**: Use distilled model
3. **Finalize**: Return results

#### Multi-Agent Routing:

The multi-agent workflow routes tasks to specialized agents:
- **QA Agent**: Answers questions
- **Summarizer**: Generates summaries
- **Validator**: Validates responses

```python
workflow = AgenticWorkflow(model_path)
result = workflow.run(query="What is machine learning?")
```

## Data Flow

```
User Query
    │
    ▼
┌─────────────────┐
│  Process Input  │
│  (Parse query)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Route to Agent │
│  (Multi-agent)  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Generate Reply  │
│ (Distilled LLM) │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Validate     │
│   (Optional)    │
└────────┬────────┘
         │
         ▼
    Response
```

## Key Design Decisions

### 1. Modular Architecture

Each layer (data, distillation, workflow) is independent and can be used separately or together.

**Benefits**:
- Easy to test individual components
- Flexible to swap implementations
- Clear separation of concerns

### 2. Configuration-Driven

All parameters are externalized in JSON config files.

**Benefits**:
- Easy to experiment with different settings
- No code changes needed for tuning
- Reproducible experiments

### 3. LangGraph for Workflows

Using LangGraph provides:
- State management
- Easy visualization
- Modular node composition
- Built-in error handling

### 4. Structured Data Focus

Focusing on structured data (CSV/JSON) enables:
- Domain-specific distillation
- Quality control of training data
- Easy integration with databases
- Clear data lineage

## Performance Considerations

### Model Size Reduction

Distillation typically reduces:
- Model size: 40-60% smaller
- Inference time: 2-3x faster
- Memory usage: 50% less

### Trade-offs

- **Accuracy**: 2-5% drop typically
- **Specialization**: Better on domain-specific tasks
- **Generalization**: May be worse on out-of-domain tasks

## Extension Points

### Adding New Data Sources

Extend `StructuredDataLoader`:

```python
def load_from_database(self, connection_string, query):
    # Custom database loading logic
    pass
```

### Custom Distillation Strategies

Extend `DistillationLoss`:

```python
class CustomDistillationLoss(DistillationLoss):
    def forward(self, student_logits, teacher_logits, labels):
        # Custom loss computation
        pass
```

### New Agent Types

Add new nodes to the workflow:

```python
def custom_agent_node(self, state: AgentState) -> AgentState:
    # Custom agent logic
    return state

workflow.add_node("custom_agent", custom_agent_node)
```

## Best Practices

### 1. Data Preparation

- **Quality over quantity**: 100 high-quality examples > 1000 noisy ones
- **Balance**: Ensure balanced representation of different classes/topics
- **Validation**: Always validate loaded data before training

### 2. Distillation

- **Temperature tuning**: Start with 2.0, adjust based on results
- **Alpha parameter**: Balance between distillation and hard labels
- **Gradual unfreezing**: Consider unfreezing student layers gradually

### 3. Workflow Design

- **State management**: Keep state minimal and well-defined
- **Error handling**: Add validation nodes for production
- **Logging**: Log all agent decisions for debugging

## Security Considerations

### Model Security

- Use patched versions of dependencies (torch >= 2.6.0, transformers >= 4.48.0)
- Validate model checkpoints before loading
- Don't load models from untrusted sources

### Data Security

- Sanitize user inputs before processing
- Don't log sensitive information
- Implement rate limiting for production deployments

## Testing Strategy

### Unit Tests

Test individual components:
- Data loading functions
- Configuration parsing
- Model initialization

### Integration Tests

Test component interactions:
- Data → Distillation pipeline
- Distilled model → Agent workflow
- End-to-end query processing

### Performance Tests

Measure:
- Distillation time
- Inference latency
- Memory usage

## Deployment

### Local Development

```bash
pip install -r requirements.txt
python examples/complete_pipeline.py
```

### Production Deployment

Consider:
- Model serving (TorchServe, ONNX Runtime)
- API wrapper (FastAPI, Flask)
- Monitoring (Prometheus, Grafana)
- Scaling (Kubernetes, Docker)

## Future Enhancements

1. **Support for more model architectures**: BERT, T5, LLaMA
2. **Advanced distillation techniques**: Progressive, multi-stage
3. **Better agent workflows**: Reinforcement learning, tool use
4. **Integration with vector databases**: For retrieval-augmented generation
5. **Model quantization**: Further compress distilled models

## References

- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) - Hinton et al.
- [DistilBERT](https://arxiv.org/abs/1910.01108) - Sanh et al.
- [LangGraph Documentation](https://python.langchain.com/docs/langgraph)
- [Transformers Library](https://huggingface.co/docs/transformers/)
