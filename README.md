# Multi-Agent Large Language Model Tournaments for Lightweight Heuristic Generation in Augmented Reality Visual Acuity Screening (MICCAI 2026)

This repository contains the official code for our MICCAI 2026 paper, which implements a tournament-based multi-agent framework to generate and evolve lightweight, explainable heuristics for estimating Visual Acuity (VA) scores in Augmented Reality.

## Prerequisites

To run this framework, you need:
- **Python 3.8+**
- **Gemini API Key**: Used by the agent for code generation and orchestrating the tournament (we use `gemini-2.5-flash`).
- **Local LLM**: An OpenAI-compatible local model API (e.g., `gpt-oss`, or `Qwen` hosted via vLLM or Ollama). The framework sends POST requests to this local model for heuristic generation.

## Setup

1. **Install Dependencies**:
   Ensure you have the required Python packages installed. You can install them using pip:
   ```bash
   pip install mcp google-generativeai pandas requests python-dotenv ipython
   ```

2. **Configure Environment Variables**:
   Create a `.env` file in your working directory and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```

3. **Configure Local Model**:
   Update the `api_url` in `tournament_framework_code/gpt_oss_np.py` to point to your local model's correct POST endpoint.

## Running the Tournament

The framework uses a client-server architecture powered by the Model Context Protocol (MCP). The server contains the core logic for the competition loops, while the client provides an intelligent agent interface.

To start the system, run the client and pass the server script as an argument:

```bash
python tournament_framework_code/competition_client.py tournament_framework_code/competition_server-tour-loop-update.py
```

### Interactive Agent Chat

Once connected, you will enter an interactive chat environment. The chat is backed by an Agentic AI (Gemini 2.5 Flash) that understands the available tools. You can instruct the agent using natural language.

**Example Commands you can type in the chat:**
- *"Start a new competition with 10 total examples and 4 contestants"*
- *"Get the current competition status"*
- *"Advance to the next round"*
- *"Run a full competition with maximum 5 rounds"*
- *"Get final results"*

The agent will automatically translate your requests into the correct tool calls and execute the tournament logic for you.

**Advanced Usage Example:**
You can specify advanced configuration parameters, including custom data splits and evaluation metrics via the chat agent. For example, explicitly pathing the metadata:

> *"Run full competition with m=8, n=48, evolution_strategy=tournament, metric_mode=RMSE, evaluate_mode=pair, prompt_mode=add_all_contestants_step, train_data='/path/to/mock_directory/train_data.csv', evaluation_data='/path/to/mock_directory/eval_data.csv', test_data='/path/to/mock_directory/test_data.csv'"*

## Flexibility & Repurposing

This framework is highly configurable and can be adapted to tasks beyond Visual Acuity. 

By default, the codebase reads a generic `metadata.xlsx` (or custom data paths) to locate the respective `.csv` files for each participant's individual trials. This data retrieval structure allows you to use the exact same LLM tournament and heuristic generation pipelines on entirely different datasets, simply by formatting your inputs to match the expected metadata and CSV structure and supplying the right scoring metric.

### Understanding the Mock Data
The original patient data consists of tabular `.csv` files tracking each individual trial. However, to preserve patient privacy, the mock data provided in this repository has had sensitive columns removed and is presented as whitespace-separated `.txt` files rather than pure CSVs. 

Both the mock data provided in `mock_data/sample_input.txt` and the actual heuristic algorithms evolved by the LLM are designed to correctly parse this whitespace-separated, stripped-down text format.
