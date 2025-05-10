# Multi-Agent Healthcare Document Flow

This project implements a multi-agent system for healthcare document processing using LangGraph and various LLM providers.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
   - Create a `.env` file in the project root directory
   - Add your API keys to the `.env` file:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   MISTRAL_API_KEY=your-mistral-api-key-here
   ```

## Usage

The system supports multiple LLM providers (OpenAI and Mistral) and can be used as follows:

```python
from multiagent_document_flow import process_patient_request
from llm_instances import ModelProvider, ModelConfig

# Using default OpenAI model
result = process_patient_request("P12345")

# Using Mistral model
mistral_config = ModelConfig(
    provider=ModelProvider.MISTRAL,
    model_name="mistral-large-latest",
    temperature=0.7
)
result = process_patient_request("P12345", mistral_config)
```

## Environment Variables

The following environment variables are required:

- `OPENAI_API_KEY`: Your OpenAI API key
- `MISTRAL_API_KEY`: Your Mistral API key

You can either:
1. Set these in your `.env` file
2. Provide them directly when creating a `ModelConfig` instance
3. Set them as system environment variables

## Project Structure

- `multiagent_document_flow.py`: Main workflow implementation
- `llm_instances.py`: LLM configuration and instantiation
- `.env`: Environment variables (create this file with your API keys)
- `requirements.txt`: Project dependencies 