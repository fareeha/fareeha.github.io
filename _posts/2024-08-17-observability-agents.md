---
title: "Observability for LLM Systems"
date: 2024-08-17
categories: [Machine Learning]
tags: [llm, observability, evaluation, agents]
---


You've built an LLM-powered app. It works great in your notebook. Then you ship it, and something breaks — but *what*? Was it the prompt? The retrieval? The model itself? Without observability, you're debugging in the dark.

This is a walkthrough of how to evaluate, trace, and monitor LLM systems in production — from model benchmarks to system-level evals to the instrumentation that ties it all together.

---

## LLM Evaluation: Two Levels

There are two very different things you might be evaluating, and it's worth being precise about which one you mean.

### Model-Level Benchmarks

These measure the *model itself* — how well the raw LLM performs on standardized tasks, independent of your application.

**Common benchmark datasets:**

- **MMLU** — massive multitask language understanding. Multiple-choice questions covering math, philosophy, medicine, law, and dozens of other domains. Tests breadth of knowledge.
- **HumanEval** — code generation benchmark. The model gets a function signature and docstring and has to produce working code. Tests programming ability.
- **TruthfulQA** — tests whether the model produces truthful answers vs. common misconceptions.
- **GSM8K** — grade school math. Simple-sounding but surprisingly hard for models — tests multi-step reasoning.

These benchmarks are useful for *choosing* a model, but they don't tell you how well that model works in *your* system. A model that tops MMLU might still hallucinate on your specific domain data.

### System-Level Evaluation

This is where it gets real. System-level eval measures how well your *entire system* performs — the LLM plus retrieval, prompts, tools, routing, and everything else — against your actual business requirements.

**Where does the test data come from?**

- **Manually created** — you write test cases by hand. High quality but expensive and slow.
- **Synthesized** — use a strong LLM to generate test cases. Fast and scalable, but needs human review.
- **Curated from production** — pull real user queries and annotate them. The gold standard because it reflects actual usage patterns, edge cases, and failure modes.

In practice, you want a mix of all three. Manual cases for known edge cases, synthetic for coverage, and production data for grounding.

---

## Types of Evaluation

Different parts of your system need different kinds of evaluation. Here are the main ones:

### Hallucination Detection

Is the model making things up? This is the big one. You can evaluate hallucinations by checking whether the model's claims are grounded in the provided context (for RAG systems) or factually accurate (for general knowledge).

```python
# Example eval rubric for hallucination scoring
hallucination_eval_prompt = """Given the following context and response,
score the response for faithfulness on a scale of 1-5:

1 = Contains fabricated facts not in the context
3 = Mostly grounded but includes minor unsupported claims
5 = Fully grounded in the provided context

Context: {context}
Response: {response}
Score:"""
```

### Retrieval Relevance

For RAG systems: are you retrieving the *right* documents? A perfect LLM can't help if it's being fed irrelevant context. Measure this by comparing retrieved documents against ground-truth relevant documents for a set of test queries.

### QA Correctness on Retrieved Data

Even if retrieval is good, the model might misinterpret or ignore the retrieved context. This evaluates whether the final answer actually uses the retrieved information correctly.

### Toxicity

Is the model producing harmful, offensive, or biased content? Especially important for user-facing applications. Tools like Perspective API can automate this at scale.

### Summarization Quality

For summarization tasks: is the summary accurate, complete, and concise? Does it capture the key points without introducing information that wasn't in the source?

### Code Correctness and Readability

For code generation: does the code run? Does it pass test cases? And beyond correctness — is it readable, well-structured, and idiomatic?

```python
# Simple code eval: run generated code against test cases
def evaluate_code_generation(generated_code, test_cases):
    results = []
    for test in test_cases:
        try:
            exec(generated_code)  # execute the generated function
            output = eval(test["call"])  # run the test input
            passed = output == test["expected"]
            results.append({"test": test["call"], "passed": passed})
        except Exception as e:
            results.append({"test": test["call"], "passed": False, "error": str(e)})
    return results
```

---

## Agents: Why Observability Gets Harder

Agents make observability significantly more complex because they don't follow a fixed path — they reason, route, and act dynamically. An agent has three main components:

**1. Reasoning** — powered by the LLM. The agent interprets the user's request, breaks it into steps, and decides what to do next. This is where most of the "intelligence" lives, and also where most failures originate.

**2. Routing** — interpreting the request and picking the correct tool. Should it search the database? Call an API? Ask a follow-up question? Routing errors are sneaky because the model might confidently pick the wrong tool and still produce a plausible-looking answer.

**3. Action** — executing code, calling tools, hitting APIs, making further LLM calls. This is where the agent interacts with the outside world, and where latency, errors, and unexpected responses can cascade.

The core challenge: **small changes to code or prompts can cause performance regression.** You tweak a system prompt, and suddenly the agent stops using a tool it used to use reliably. You update a dependency, and retrieval latency doubles. These regressions are subtle and often don't surface until users complain.

**The solution:** maintain a representative set of test cases and datasets that cover your key use cases. Run them on every change. Treat your eval suite the way a software engineer treats their test suite — if it doesn't pass, it doesn't ship.

---

## Observability: Seeing Inside Your LLM App

Observability means having **complete visibility into every layer** of your LLM-based application — the app logic, the prompts going in, the responses coming out, the tools being called, and everything in between.

Without it, your LLM app is a black box. With it, you can answer questions like: *Why did the model hallucinate on this query? Which retrieval step returned bad context? Why did this request take 12 seconds?*

### The Building Blocks: Traces and Spans

The two fundamental concepts are borrowed from distributed systems observability:

**Traces** — a trace represents one complete run through your application. A user asks a question, your app processes it, and a response comes back — that entire journey is one trace.

**Spans** — a span captures data about an *individual step* within a trace. Each span records what happened, how long it took, what the inputs and outputs were, and whether it succeeded or failed.

A single trace is made up of multiple nested spans. For example, a RAG chatbot query might produce a trace like this:

```
Trace: "What were Q3 sales for ACME Corp?"
│
├── Span: embedding_generation (12ms)
│   └── Input: user query → Output: 384-dim vector
│
├── Span: vector_search (45ms)
│   └── Input: query embedding → Output: 5 retrieved documents
│
├── Span: prompt_assembly (2ms)
│   └── Input: query + retrieved docs → Output: formatted prompt (1,847 tokens)
│
├── Span: llm_completion (1,203ms)
│   └── Input: prompt → Output: response (312 tokens)
│   └── Model: gpt-4, temperature: 0.1, cost: $0.04
│
└── Span: response_postprocessing (5ms)
    └── Input: raw response → Output: formatted answer
```

At a glance, you can see exactly where time is being spent, what data is flowing between steps, and where things might be going wrong.

### Common Span Types in LLM Apps

Different steps in your app produce different types of spans:

- **LLM Span** (ChatCompletion) — captures the prompt, response, model used, token count, latency, and cost for each LLM call
- **Chain Span** — captures higher-level orchestration like router decisions, tool selection, and agent reasoning steps
- **Tool Span** — captures individual tool executions, like "lookup sales data" or "query database," including inputs, outputs, and errors
- **Retrieval Span** — captures vector search or document retrieval, including the query, results, and relevance scores
- **Embedding Span** — captures embedding generation for queries or documents

### OpenTelemetry (OTEL)

**OpenTelemetry** is the industry-standard framework for observability, and it's increasingly the default for LLM apps too. It provides a vendor-neutral way to instrument your code and export traces to whatever backend you're using (Arize, Datadog, Honeycomb, etc.).

The basic pattern is wrapping your code in spans:

```python
from opentelemetry import trace

# Get a tracer instance for your application
tracer = trace.get_tracer("my-llm-app")

def process_query(user_query):
    # Create a parent span for the entire query processing
    with tracer.start_as_current_span("process_query") as span:
        # Record the input as a span attribute
        span.set_attribute("user.query", user_query)
        
        # Child span for retrieval
        with tracer.start_as_current_span("retrieve_context") as retrieval_span:
            docs = vector_store.search(user_query, top_k=5)
            retrieval_span.set_attribute("retrieval.num_docs", len(docs))
            retrieval_span.set_attribute("retrieval.top_score", docs[0].score)
        
        # Child span for LLM completion
        with tracer.start_as_current_span("llm_completion") as llm_span:
            response = llm.chat(prompt=build_prompt(user_query, docs))
            llm_span.set_attribute("llm.model", "gpt-4")
            llm_span.set_attribute("llm.tokens.input", response.usage.prompt_tokens)
            llm_span.set_attribute("llm.tokens.output", response.usage.completion_tokens)
        
        return response.content
```

You can also use **decorators** for cleaner instrumentation when you don't need to set attributes mid-function:

```python
from opentelemetry.instrumentation import instrument

# Decorator approach — less boilerplate, still captures timing and errors
@instrument(tracer, "retrieve_context")
def retrieve_context(query, top_k=5):
    return vector_store.search(query, top_k=top_k)

@instrument(tracer, "generate_response")
def generate_response(query, context):
    prompt = build_prompt(query, context)
    return llm.chat(prompt=prompt)
```

Platforms like **Arize**, **LangSmith**, and **Weights & Biases** provide auto-instrumentation that wraps common libraries (OpenAI, LangChain, LlamaIndex) so you get spans automatically without manually adding `with` blocks everywhere.

### Why Do This?

If you're wondering whether all this instrumentation is worth the effort, here's what it gives you:

**Debugging** — when a user reports a bad response, you can pull up the exact trace, see the exact prompt that was sent, the exact context that was retrieved, and pinpoint where things went wrong. No more guessing.

**Performance monitoring** — track latency, token usage, and cost across every step. Find bottlenecks (is retrieval slow? is the model slow?) and optimize where it matters.

**Regression detection** — compare trace patterns before and after a change. Did your prompt update cause the model to use more tokens? Did retrieval relevance drop?

**Quality tracking** — attach eval scores to traces. Over time, you build a picture of how your system's quality is trending — across models, prompt versions, and user segments.

**Cost management** — LLM calls aren't free. Tracing lets you see exactly where your money is going and identify opportunities to cache, batch, or route to cheaper models.

---

## Putting It All Together

A production-grade LLM observability setup looks something like this:

1. **Instrument your app** with OpenTelemetry spans at each meaningful step
2. **Export traces** to an observability platform (Arize, LangSmith, Datadog, etc.)
3. **Maintain an eval suite** — representative test cases that run on every deploy
4. **Attach eval scores to traces** — so you can correlate quality with system behavior
5. **Set up alerts** — on latency spikes, cost anomalies, hallucination rate increases, or retrieval relevance drops
6. **Review production traces regularly** — not just when things break, but to understand how your system behaves in the wild

The goal isn't perfection — it's visibility. You can't improve what you can't see.

---

## Recap

- **Model benchmarks** (MMLU, HumanEval) help you *choose* a model; **system evals** tell you if your app actually works
- **Eval types** span hallucination, retrieval relevance, toxicity, summarization, and code quality — pick the ones that matter for your use case
- **Agents** make observability harder because they reason, route, and act dynamically — small changes can cause regressions
- **Traces and spans** are the building blocks — a trace is one full request, spans are individual steps within it
- **OpenTelemetry** is the standard for instrumentation — use `with` blocks or decorators to capture spans
- **The payoff** is debugging, performance monitoring, regression detection, and cost management — all from the same instrumentation

Ship it, trace it, eval it, improve it. That's the loop.
