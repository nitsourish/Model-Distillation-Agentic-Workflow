# Model-Distillation-Agentic-Workflow

Model Distillation utilizing structured data and consuming distilled models using Agentic Workflow in LangGraph. This project demonstrates knowledge distillation techniques using TabPFN (Tabular Prior-Data Fitted Networks) as the teacher model and simpler neural networks as student models.

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Setting up the Environment

This project uses `uv` for fast Python package management and virtual environment creation.

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd model_distillation
   ```

3. **Create and activate virtual environment with uv**:
   ```bash
   # Create virtual environment and install dependencies
   uv sync
   
   # Activate the virtual environment
   source .venv/bin/activate
   ```

4. **Verify installation**:
   ```bash
   python --version
   uv pip list
   ```

## 📊 Project Structure

```
model_distillation/
├── pyproject.toml            # Project configuration and dependencies
├── uv.lock                   # Locked dependency versions
├── notebooks/                # Jupyter notebooks for experimentation
│   ├── data_prep_exploreTabPFN_foundation_model.ipynb
│   └── model_distillation.ipynb
├── artefacts/               # Generated models and outputs
└── data/                    # Dataset files
```

## 📓 Jupyter Notebooks Overview

### 1. Data Preparation and TabPFN Exploration
**File:** `notebooks/data_prep_exploreTabPFN_foundation_model.ipynb`

This notebook focuses on:
- **Data Engineering**: Creating a multi-output regression task using the Adult dataset from fairlearn
- **Synthetic Target Creation**: Generates a continuous target variable `max_loan` based on sophisticated business rules considering:
  - Age demographics and risk assessment
  - Employment sector preferences (Federal-gov > State-gov > Private > Self-employed)
  - Education level correlation with loan eligibility
  - Occupation-based risk evaluation
  - Financial indicators (capital gains/losses)
  - Work patterns and their correlation with income potential
- **TabPFN Foundation Model**: Explores the capabilities of TabPFN as a teacher model for both classification and regression tasks

### 2. Model Distillation Implementation
**File:** `notebooks/model_distillation.ipynb`

This notebook implements:
- **Teacher-Student Framework**: Uses TabPFN as the teacher model to train simpler student networks
- **Multi-task Learning**: Handles both classification (loan approval) and regression (loan amount) tasks
- **Knowledge Transfer**: Demonstrates various distillation techniques including:
  - Soft target distillation
  - Feature-level knowledge transfer
  - Performance comparison between teacher and student models
- **Neural Network Architecture**: Custom PyTorch implementations for student models
- **Evaluation Metrics**: Comprehensive model evaluation using accuracy, AUC, R², and RMSE

## 🔧 Dependencies

Key dependencies managed through `pyproject.toml`:
- **TabPFN** (v2.2.1+): Prior-Data Fitted Networks for tabular data
- **PyTorch**: Deep learning framework for student model implementation
- **scikit-learn**: Traditional ML utilities and metrics
- **pandas & numpy**: Data manipulation and numerical computing
- **fairlearn**: Ethical AI and bias mitigation tools
- **MCP**: Model Context Protocol for agentic workflows

## 🎯 Usage

1. **Data Preparation**:
   ```bash
   # Run the data preparation notebook
   jupyter notebook notebooks/data_prep_exploreTabPFN_foundation_model.ipynb
   ```

2. **Model Distillation**:
   ```bash
   # Run the model distillation experiment
   jupyter notebook notebooks/model_distillation.ipynb
   ```

3. **Main Application**:
   ```bash
   python main.py
   ```

## 🔬 Key Features

- **Advanced Data Synthesis**: Creates realistic synthetic targets using domain knowledge
- **State-of-the-art Teacher Models**: Leverages TabPFN's transformer-based architecture
- **Efficient Student Models**: Trains lightweight models that retain teacher performance
- **Multi-objective Optimization**: Handles both classification and regression simultaneously
- **Ethical AI Integration**: Uses fairlearn for bias detection and mitigation
- **Reproducible Experiments**: Seed-controlled random processes for consistent results

## 📈 Model Performance

The distillation process aims to:
- Maintain >95% of teacher model accuracy in student models
- Reduce model size by 10-100x compared to the teacher
- Enable faster inference while preserving decision quality
- Provide interpretable student models for production deployment

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and ensure tests pass
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

## 📄 License

This project is licensed under the terms specified in the LICENSE file.
