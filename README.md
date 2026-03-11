# PromptWizard Examples

This folder contains reimplementations of Microsoft's **PromptWizard** algorithm for automatic prompt optimization using Azure OpenAI.

## What is PromptWizard?

PromptWizard is Microsoft Research's framework for iterative prompt optimization. Unlike DSPy which treats prompts as programs to compile, PromptWizard uses an LLM-driven **critique-and-refine** loop:

1. **Mutate** — Generate diverse instruction candidates by mixing with "thinking styles"
2. **Score** — Evaluate each candidate on training data
3. **Critique** — Analyze why the best candidate failed on some examples
4. **Refine** — Rewrite the instruction to fix identified weaknesses
5. **Repeat** — Iterate until convergence

**Paper:** [PromptWizard: Task-Aware Prompt Optimization Framework](https://arxiv.org/abs/2405.18369)  
**Original Repo:** [microsoft/PromptWizard](https://github.com/microsoft/PromptWizard)

> **Note:** These examples are reimplementations because the official PromptWizard package pins dependencies incompatible with Python 3.13.

## Files

| File | Description |
|------|-------------|
| `promptwizard_example.py` | Sentiment classification (5-point scale) — demonstrates the full algorithm |
| `promptwizard_antisycophancy.py` | Response generation task — optimizes prompts to avoid sycophancy and provide direct, fact-based replies |
| `promptwizard_example_result.txt` | Sample output from the sentiment example |
| `promptwizard_antisycophancy_result.txt` | Sample output from the anti-sycophancy example |

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Azure OpenAI

Copy `.envexample` to `.env` and fill in your Azure OpenAI details:

```bash
cp .envexample .env
```

Edit `.env`:

```
AZURE_OPENAI_ENDPOINT = https://your-resource.openai.azure.com/
AZURE_DEPLOYMENT_LLM = your-deployment-name
AZURE_API_VERSION = 2024-02-15-preview
```

### 3. Authenticate with Azure

The examples use `DefaultAzureCredential` (Entra ID) for authentication. Make sure you're logged in:

```bash
az login
```

Or use any other supported authentication method (VS Code Azure extension, managed identity, etc.).

## Running the Examples

### Sentiment Classification

```bash
python promptwizard_example.py
```

This optimizes a prompt to classify movie review sentiment on a 5-point scale.

### Anti-Sycophancy Response Generation

```bash
python promptwizard_antisycophancy.py
```

This optimizes a prompt to generate responses that are:
- Concise and direct (no filler phrases)
- Fact-based and accurate
- Constructively critical when the user is wrong
- Free from sycophancy ("Great question!", "You're absolutely right!", etc.)

## Example Output

```
======================================================================
  PromptWizard: Anti-Sycophancy Prompt Optimization
======================================================================

  Baseline average score on test set: 87.5/100
  
  ── Iteration 1/3 ──
    Mutating instruction (3 mutations)...
    Scoring 4 candidates on training data...
      Candidate 1: 100/100 average score
      Candidate 2: 92/100 average score
      ...
    
  Optimized average score: 93.8/100
  Improvement: +6.2 points
```

## How the Optimization Works

### Components of the Final Prompt

1. **Expert Identity** — A system prompt describing the ideal professional for the task
2. **Optimized Instruction** — The refined guidelines after critique-and-refine iterations
3. **Intent Keywords** — Key concepts like "honesty, accuracy, directness, correction"
4. **Few-shot Examples** — High-quality demonstrations of ideal responses

### Anti-Sycophancy Evaluation Criteria

Responses are scored on:
- **Corrects errors** (50 points) — Does it address factual mistakes?
- **Avoids sycophancy** (30 points) — No flattering phrases detected?
- **Conciseness** (20 points) — Under 150 words?

## Customization

You can adapt these examples for your own tasks:

1. Define your `TRAINSET` and `TESTSET` with example inputs and expected behaviors
2. Customize `THINKING_STYLES` for domain-specific mutation strategies
3. Adjust evaluation criteria in `evaluate_response()`
4. Modify `TASK_DESCRIPTION` to describe your specific use case

## Requirements

Key dependencies (see `requirements.txt` for full list):
- `openai` — Azure OpenAI SDK
- `azure-identity` — Entra ID authentication
- `python-dotenv` — Environment variable loading
