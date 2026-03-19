---
title: "Finetuning Bascis"
date: 2022-01-14
categories: [Machine Learning]
tags: [finetuning, python]
---

## What is Fine-Tuning?

Fine-tuning is the process of adapting a general-purpose model like GPT-3 to better suit specific tasks or domains. 

For example, fine-tuning GPT-3 to ChatGPT, optimized for conversational use. Or, GPT4 to CoPilot.

### Why Fine-Tune?

- **Goes beyond the prompt:** Injects far more data than a prompt allows—the model *learns* from it.
- **Domain expertise:** Enables consistent, reliable outputs in specific areas like healthcare, finance, or customer support.
- **Custom behavior:** Helps control tone, verbosity, formatting, and preferred styles.
- **Better control over hallucinations and outdated knowledge.**

---

### Prompting vs. Fine-Tuning

| Feature                | Prompting                                 | Fine-Tuning                                      |
|------------------------|-------------------------------------------|--------------------------------------------------|
| **Getting started**    | No data or setup needed                   | Requires curated high-quality data              |
| **Cost**               | Lower upfront cost                        | Higher initial cost. Cheaper at scale            |
| **Tech skills**        | No technical knowledge needed             | Requires ML + data expertise                     |
| **Data capacity**      | Limited to prompt size                    | No limit             |
| **Memory**             | Forgets data after use                    | Learns and remembers domain-specific knowledge   |
| **Reliability**        | May hallucinate     | More consistent and accurate                     |
| **RAG compatibility**  | Yes                                       | Yes – can combine both for best of both worlds   |
| **Best for**           | Prototypes, generic tasks                 | Enterprise, production, domain-specific models   |

---

Summary :
- Use **prompting** for fast iteration and light personalization. 
- Use **fine-tuning** when you need consistency, control, and privacy at scale. Can also reduce cost + low latency to finetune a smaller model for scale.


Libraries to use:
- PyTorch (Meta)
- HuggingFace
- Lamini (Llama library)
more ways and libraries to do it 


    
## Types of Tasks Ideal for Fine-Tuning

Fine-tuning works best on clearly defined input–output patterns. Two broad categories:

| Type        | Description                           | Examples                                          |
|-------------|---------------------------------------|--------------------------------------------------|
| **Extraction ("Reading")** | Text in → concise output | Planning steps, reasoning, tool usage in agents  |
| **Expansion ("Writing")**  | Text in → extended output | Chat, code generation, writing emails            |

Focus on one task at a time and gather high-quality examples -- it’s the biggest unlock to better model behavior.

---

## Getting Started with Fine-Tuning

A quick 5-step guide:

1. **Prompt first**  
   Try out tasks using a large LLM and see what it does *okay*.

2. **Spot weak areas**  
   Look for tasks it performs decently but not great.

3. **Pick one task**  
   Focus on a single use case.

4. **Gather data**  
   Collect ~1,000 input–output examples that improve on the LLM's output.

5. **Fine-tune small model**  
   Train a smaller LLM on your dataset for better, consistent results.

---

## Structuring Your Data for Fine-Tuning

Before fine-tuning, make sure your dataset is consistent and formatted for your prompt style.

### Step 1: Inspect the Data Format

```python
examples = instruction_dataset_df.to_dict()

if "question" in examples and "answer" in examples:
    text = examples["question"][0] + examples["answer"][0]
elif "instruction" in examples and "response" in examples:
    text = examples["instruction"][0] + examples["response"][0]
elif "input" in examples and "output" in examples:
    text = examples["input"][0] + examples["output"][0]
else:
    text = examples["text"][0]

print(text)
```

### Step 2: Create a Prompt Template

```python
prompt_template_qa = """### Question:
{question}

### Answer:
{answer}"""

question = examples["question"][0]
answer = examples["answer"][0]

text_with_prompt_template = prompt_template_qa.format(
    question=question, 
    answer=answer
)

print(text_with_prompt_template)
```

> 💡 **Why use `###` in training prompts?**
>
> - **Clear separation**: Distinguishes parts of the input (e.g., `### Instruction:` vs. `### Response:`), making structure easier for the model to learn.
> - **Helps the model parse intent**: LLMs rely on patterns. These markers serve as cues to identify what’s a question, what’s an answer, etc.
> - **Consistency during inference**: Matching the same format at train and test time helps the model associate headers with expected outputs.
> - **Avoids confusion with regular text**: `###` is rarely used in natural conversation, reducing ambiguity during learning.


```python
num_examples = len(examples["question"])
finetuning_dataset_text_only = []
finetuning_dataset_question_answer = []
for i in range(num_examples):
  question = examples["question"][i]
  answer = examples["answer"][i]

  text_with_prompt_template_qa = prompt_template_qa.format(question=question, answer=answer)
  finetuning_dataset_text_only.append({"text": text_with_prompt_template_qa})

  text_with_prompt_template_q = prompt_template_q.format(question=question)
  finetuning_dataset_question_answer.append({"question": text_with_prompt_template_q, "answer": answer})
```

Common way to store in jsonl format

```python
with jsonlines.open(f'lamini_docs_processed.jsonl', 'w') as writer:
    writer.write_all(finetuning_dataset_question_answer)
```

## Instruction Fine-Tuning

Instruction fine-tuning is a specific type of fine-tuning that teaches a model how to **follow instructions and behave more like a helpful assistant or chatbot**. It's one of the main reasons large models like ChatGPT became so widely adopted.

### What data can you use?

You likely already have valuable data:
- Company FAQs
- Customer support transcripts
- Slack or internal team conversations

These can be formatted into instruction–response pairs for fine-tuning.

### No data? No problem.

If you don’t have structured data:
- Use an LLM (like ChatGPT) to generate instruction–response pairs from existing documents
- Try prompt templates or tools like **Self-Instruct** or **Alpaca-style** methods to convert raw text into QA datasets

> Instruction fine-tuning helps models become more interactive, useful, and aligned with your domain-specific needs.


## Choosing the Right Data for Fine-Tuning

Not all data is created equal. The quality and type of data you use directly impacts how well your model learns and generalizes.

### ✅ Better Data

- **Higher quality**: Human-written, accurate, and clearly labeled
- **Diverse**: Covers a range of situations, tones, and edge cases
- **Real**: Comes from actual usage (e.g. customer support logs, human-written FAQs)
- **More**: Volume helps—but only if the quality holds up

### ⚠️ Worse Data

- **Lower quality**: Incomplete, incorrect, or unclear examples
- **Homogeneous**: Repetitive phrasing or narrow scenarios
- **Synthetic**: Fully generated without human refinement
- **Less**: Too few examples limit learning and generalization

> 💡 Aim for high-quality, real-world, and varied examples—even a few hundred good ones can outperform thousands of noisy ones.

## Preparing Your Data for Fine-Tuning

Once you've selected high-quality data, here's how to get it ready for training:

### 🛠️ Steps to Prepare Your Dataset

1. **Collect instruction–response pairs**  
   Use real examples or generate them with an LLM.

2. **Concatenate pairs**  
   Format them into a single prompt using a template (e.g., `### Instruction:` and `### Response:`).

3. **Tokenize**  
   Convert text into tokens. Apply padding or truncation as needed to fit model limits.

4. **Split the dataset**  
   Divide your data into training and test sets to evaluate model performance.

> 🔁 Tip: Always validate a few samples after each step—bad formatting or token overflow can silently break your training run.

---

## Types of Fine-Tuning

So far we've talked about fine-tuning as one thing, but there are actually different approaches depending on how much of the model you want to update.

### Full Fine-Tuning

This is the traditional approach — you update *every* weight in the model. Instruction fine-tuning (which we covered above) is a common form of full fine-tuning. You take a pretrained model and train it end-to-end on your task-specific dataset.

**Pros:**

- Maximum flexibility — the model can fully adapt to your domain
- Best performance when you have enough high-quality data

**Cons:**

- Expensive — requires significant compute (GPUs, memory, time)
- Catastrophic forgetting — the model can lose its general capabilities if you're not careful
- Produces a full copy of the model weights, which is costly to store and serve

### Parameter-Efficient Fine-Tuning (PEFT)

What if you could get most of the benefit of fine-tuning while only updating a tiny fraction of the model's parameters? That's the idea behind PEFT methods.

The most popular one is **LoRA (Low-Rank Adaptation)**. Instead of updating all the weights, LoRA freezes the original model and injects small, trainable "adapter" matrices into specific layers. During training, only these adapters are updated — the base model stays untouched.

```python
from peft import LoraConfig, get_peft_model

# Configure LoRA: r is the rank (lower = fewer params), alpha scales the adapters
lora_config = LoraConfig(
    r=16,                        # rank of the adapter matrices
    lora_alpha=32,               # scaling factor for the adapter weights
    target_modules=["q_proj", "v_proj"],  # which layers to inject adapters into
    lora_dropout=0.1,            # dropout on adapter layers for regularization
    bias="none",
    task_type="CAUSAL_LM"        # causal language model (e.g., GPT-style)
)

# Wrap the base model with LoRA adapters — only adapter params are trainable
model = get_peft_model(base_model, lora_config)

# Check how few parameters we're actually training
model.print_trainable_parameters()
# Typical output: "trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.06%"
```

**Why LoRA is a big deal:**

- You're training ~0.1% of the parameters instead of 100%
- The base model stays frozen, so no catastrophic forgetting
- Adapter weights are tiny (often just a few MB) — easy to swap, store, and serve multiple fine-tunes from one base model
- Training is dramatically faster and cheaper

### QLoRA: LoRA + Quantization

QLoRA takes this even further by quantizing the base model to 4-bit precision before applying LoRA adapters. This means you can fine-tune a 65B parameter model on a single GPU — something that would normally require a cluster.

```python
from transformers import BitsAndBytesConfig

# Quantize the base model to 4-bit to slash memory usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # normalized float 4 — optimized for LLMs
    bnb_4bit_compute_dtype=torch.float16, # compute in fp16 for speed
    bnb_4bit_use_double_quant=True        # quantize the quantization constants too
)

# Load the model in 4-bit, then apply LoRA on top
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto"
)
model = get_peft_model(model, lora_config)
```

### When to Use What

| Approach | Params Updated | Compute | Best For |
|---|---|---|---|
| Full Fine-Tuning | All | High (multi-GPU) | Maximum performance, large budgets |
| LoRA | ~0.1% | Low (single GPU) | Most production use cases |
| QLoRA | ~0.1% (4-bit base) | Very low | Large models on limited hardware |

For most people starting out, **LoRA is the sweet spot** — you get 90%+ of the performance of full fine-tuning at a fraction of the cost.

---

## Recap

- **Prompting** is your starting point — use it to prototype and identify gaps
- **Fine-tuning** fills those gaps with consistent, domain-specific behavior
- **Instruction fine-tuning** is how models learn to be helpful assistants
- **Data quality > quantity** — a few hundred great examples beat thousands of noisy ones
- **LoRA/QLoRA** make fine-tuning accessible without a massive compute budget

The best approach is usually iterative: prompt first, spot the weaknesses, gather targeted data, fine-tune a small model, and ship it.

Happy fine-tuning!
