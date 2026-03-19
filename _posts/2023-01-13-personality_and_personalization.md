---
title: "Personality Post-Training"
date: 2023-01-13
categories: [Machine Learning]
tags: [transformers, personlization]
---

How do you take a base LLM and give it a *personality*? Not just make it accurate — make it funny, empathetic, blunt, or sound like a specific character? That's what post-training is for.

This is a walkthrough of the post-training stack: SFT, RLHF, and DPO — what each does, when to use which, and how they connect to the bigger challenge of personality and personalization in LLMs.

---

## 🧩 The LLM Post-Training Stack

A pretrained LLM knows a lot, but it doesn't know how to *behave*. Post-training is the set of techniques that take a raw model and turn it into something you'd actually want to talk to. There are three main stages, and they build on each other.

### 1. Supervised Fine-Tuning (SFT)

SFT is the most straightforward step: you fine-tune the model on curated prompt–response pairs that demonstrate the behavior you want. This is where you instill structure — how to follow instructions, when to ask clarifying questions, how to format a response.

Think of it as teaching by example. You show the model hundreds or thousands of conversations that represent *good* behavior, and the model learns to mimic those patterns.

```python
# SFT training data is just prompt-completion pairs
# The model learns to produce outputs that look like your examples
sft_examples = [
    {
        "prompt": "Explain quantum computing to a 10-year-old.",
        "completion": "Imagine a regular computer is like a light switch — it's either on or off. A quantum computer is like a dimmer switch that can be on, off, or anywhere in between, all at the same time..."
    },
    {
        "prompt": "Write a professional email declining a meeting.",
        "completion": "Hi [Name],\n\nThanks for the invite. Unfortunately I won't be able to make it this week..."
    }
]
```

SFT is great at teaching *what* to say, but it's less great at teaching *how* to say it — the subtle stuff like tone, humor, and nuance. For that, you need preference-based training.

### 2. RLHF (Reinforcement Learning from Human Feedback)

RLHF is where things get more sophisticated. Instead of showing the model correct outputs, you show it *pairs* of outputs and have humans label which one is better. From that preference data, you train a **reward model** — a separate model that learns to score how "good" a response is.

Then you use **PPO (Proximal Policy Optimization)** to optimize your LLM to generate responses that score higher according to that reward model.

**What you need:**

- A **preference dataset** — pairs of responses where humans labeled A vs. B
- A **prompt dataset** — prompts for the PPO training loop to generate responses against

```python
# Preference data: same prompt, two responses, human picks the better one
preference_data = [
    {
        "prompt": "Tell me a joke about programming.",
        "chosen": "Why do programmers prefer dark mode? Because light attracts bugs.",
        "rejected": "Here is a joke: Programming is hard. That's the joke."
    }
]

# Step 1: Train a reward model on these preferences
# Step 2: Use PPO to optimize the LLM against the reward model
from trl import PPOTrainer, PPOConfig

ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=16,
    mini_batch_size=4,
    ppo_epochs=4,                # number of PPO update passes per batch
    kl_penalty="kl",
    init_kl_coef=0.2,           # KL coefficient — controls how far the model can drift
)
```

#### The Reward Hacking Problem

Here's the tricky part. The reward model is an *approximation* of human preferences, and the LLM will find ways to exploit it. This is called **reward hacking**.

Classic example: if your reward model learned that humans prefer positive, encouraging responses, the LLM might start producing absurdly over-the-top positivity for *every* query — even when the situation calls for honesty or nuance. The reward score goes up, but the actual quality goes down.

The **KL coefficient** (`init_kl_coef`) is the main defense against this. It penalizes the model for straying too far from its original behavior (the SFT checkpoint). Think of it as a leash:

- **Too low** → the model wanders far from the base, reward hacking becomes likely
- **Too high** → the model barely changes, you don't get the benefit of RLHF
- **Sweet spot** → usually somewhere between 0.1 and 0.3, but you'll need to tune it

Other things that act as multipliers on reward hacking risk: the quality of your preference dataset (noisy labels = noisy reward model), how many PPO steps you run (more steps = more room to overfit), and how expressive the reward model is (too simple = easy to exploit).

### 3. DPO (Direct Preference Optimization)

DPO is a newer approach that cuts out the reward model and PPO loop entirely. Instead of training a separate reward model and then doing RL, DPO uses the preference pairs *directly* to update the language model.

The key insight: you can reformulate the RLHF objective so that the language model itself implicitly becomes the reward model. Same preference data in, better-behaved model out — but with way less infrastructure.

```python
from trl import DPOTrainer, DPOConfig

dpo_config = DPOConfig(
    learning_rate=5e-7,
    beta=0.1,                    # controls strength of preference optimization
    batch_size=4,
    max_length=512,
    max_prompt_length=256,
)

# DPO trainer takes the preference pairs directly — no reward model needed
trainer = DPOTrainer(
    model=model,
    ref_model=ref_model,         # frozen copy of the base model (acts as the KL anchor)
    args=dpo_config,
    train_dataset=preference_data,
    tokenizer=tokenizer,
)

trainer.train()
```

**Why DPO is gaining traction:**

- Simpler pipeline — no reward model to train and maintain
- No PPO instabilities (PPO can be finicky to get right)
- Often matches or exceeds RLHF performance in practice
- Fewer hyperparameters to tune

The tradeoff: DPO is less flexible than RLHF when you want to optimize for multiple reward signals simultaneously, and it can be sensitive to the quality of your preference pairs.

### When to Use What

| Technique | Best For | Complexity |
|---|---|---|
| **SFT** | Structure, instruction following, formatting, basic alignment | Low |
| **RLHF** | Nuanced tone, humor, complex multi-objective optimization | High |
| **DPO** | Same goals as RLHF but with simpler infrastructure | Medium |

In practice, most production models use **SFT first, then RLHF or DPO on top**. SFT gets you 80% of the way there; preference optimization adds the polish.

---

## 💬 Personality & Personalization

This is where it gets fun. Post-training isn't just about making a model helpful — it's about giving it a *voice*. Here are the main ways to encode personality into an LLM:

### Prompt Engineering (System Prompt)

The simplest approach. You define the persona in the system prompt and the model adapts at inference time. No training required — but the personality is shallow and can break under pressure.

```python
system_prompt = """You are a witty, slightly sarcastic assistant who explains 
complex topics using pop culture references. You never use corporate jargon. 
When you don't know something, you say so with humor."""
```

**Pros:** Zero cost, instant iteration, easy to A/B test different personas.
**Cons:** The model can "forget" the persona in long conversations, tone is inconsistent, and you're using up context window on instructions instead of content.

### Fine-Tuning on Persona-Rich Dialogues

For deeper personality, you fine-tune on dialogues that *embody* the personality you want. This could be conversations in the style of a fictional character, a celebrity's interview transcripts, or synthetic dialogues generated to match a specific tone.

```python
# Fine-tuning data for a warm, encouraging persona
persona_data = [
    {
        "prompt": "I failed my exam again.",
        "completion": "Okay, that stings — I get it. But let's not spiral. What topic tripped you up the most? Sometimes the thing that feels hardest is actually the closest to clicking."
    },
    {
        "prompt": "Should I quit my job to start a company?",
        "completion": "That's a big one. I'm not going to give you a yes or no because honestly, both paths are valid. But here's what I'd think about: do you have 6 months of runway? Is the idea keeping you up at night in a good way? And — this is the real one — are you running toward something or away from something?"
    }
]
```

The model doesn't just learn *what* to say — it learns the *rhythm* of the personality. Short sentences vs. long. Questions vs. statements. When to be direct vs. when to be gentle.

### Adapter Layers / LoRA for Persona Modules

This is the most elegant approach for serving multiple personalities from one base model. You train a separate LoRA adapter for each persona, and swap them in at inference time.

```python
from peft import LoraConfig

# Each persona gets its own lightweight adapter
playful_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
empathetic_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
blunt_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])

# Train each adapter on persona-specific dialogue data
# At inference: load the base model once, swap adapters per request
# One base model, many personalities — each adapter is just a few MB
```

This lets you run one base model in production and serve wildly different personas just by swapping a tiny adapter file. Efficient, modular, and easy to iterate on.

---

## 🧪 Evaluation Systems

Building personality is one thing — *measuring* it is another. You need both human and automated evaluation, and they measure different things.

### Human Evaluation

Still the gold standard for personality and tone. Three common approaches:

**Pairwise comparisons** — show evaluators two responses and ask "which one sounds more like the persona?" This is the most reliable signal and directly maps to the preference data format used in RLHF/DPO.

**Likert scales** — rate responses on a 1–5 scale for specific traits: helpfulness, warmth, humor, coherence, etc. Gives you more granular signal but is noisier and more subjective.

**Trait annotation** — evaluators tag whether specific traits are present. "Does this feel playful?" "Is the model being genuinely empathetic or performing empathy?" These binary/ternary judgments are cheaper to collect than Likert ratings and often more actionable.

### Synthetic Evaluation

For fast iteration, use a stronger model to judge your model's outputs. This is increasingly common and works surprisingly well for personality consistency.

```python
# Use a strong model as a judge with a structured rubric
eval_prompt = """Rate the following response on these dimensions (1-5):
- Persona consistency: Does it match the defined personality?
- Tone appropriateness: Is the tone right for the context?
- Helpfulness: Does it actually answer the question?
- Naturalness: Does it sound human or robotic?

Persona definition: {persona_description}
User message: {user_message}
Model response: {model_response}

Return scores as JSON."""
```

You can also use specialized auto-scorers for things like toxicity (Perspective API), coherence, and factual accuracy — these complement personality evaluation rather than replace it.

### Key Metrics to Track

- **Persona consistency** — does the model maintain character across different topics and conversation lengths?
- **Behavior under pressure** — what happens during moral dilemmas, adversarial prompts, or emotionally charged conversations? This is where personality tends to break.
- **Helpfulness** — personality means nothing if the model stops being useful
- **Safety** — a funny persona still can't produce harmful content
- **Diversity** — is the model generating varied responses, or falling into repetitive patterns?
- **Latency** — personality features can't come at the cost of response time

A useful framework from psychology: the **Big Five (OCEAN)** personality traits — Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism. You can design evaluation probes around these dimensions to systematically test whether your model's personality is coherent and stable. Research groups have shown that LLMs do exhibit measurable personality signatures on these scales, and that post-training reliably shifts them.

---

## 📊 Scalable Personalization Systems

Once you've nailed one persona, the next question is: how do you personalize at scale? How do you make the model adapt to *each user* over time?

### User Modeling

The goal is to build a persistent representation of each user's preferences, communication style, and context — and inject that into the model at inference time.

**Vector embeddings for user memory** — encode past interactions, stated preferences, and behavioral signals into a user embedding vector. At inference time, this vector gets prepended or injected into the model's context to steer its behavior.

**Token-based memory injection** — compress user preferences into a structured block of tokens that gets added to the system prompt. Something like: "This user prefers concise answers, responds well to humor, is an expert in ML, dislikes corporate speak."

**Personal context documents (RAG-style)** — maintain a document store per user with their conversation history, preferences, and key facts. Retrieve relevant context at query time and inject it into the prompt. This is essentially RAG but for personalization instead of knowledge retrieval.

### Data Generation for Personas

When you need training data for specific personas but don't have it organically:

**Simulate dialogues** — prompt an LLM to generate realistic conversations in specific styles. Vary across dimensions: sarcastic vs. stoic, optimistic vs. realistic, verbose vs. terse. Use these as SFT or DPO training data.

**Style transfer at scale** — take existing high-quality conversations and rewrite them in different voices. One source dialogue can produce five persona variants, multiplying your training data efficiently.

```python
# Generate persona-specific training data from a strong model
generation_prompt = """Rewrite the following assistant response in the voice of 
a {persona_style} assistant. Keep the factual content identical but change the 
tone, word choice, and sentence structure to match the persona.

Original: {original_response}
Rewritten ({persona_style}):"""

persona_styles = ["warm and encouraging", "dry and witty", "direct and no-nonsense",
                  "curious and socratic", "calm and grounding"]
```

The key insight: personality isn't one thing. It's a combination of tone, pacing, vocabulary, emotional range, and conversational habits. The more precisely you can define and measure those dimensions, the more reliably you can train for them.

---

## Recap

- **SFT** teaches the model what to say — structure, formatting, instruction following
- **RLHF** teaches it *how* to say it — tone, nuance, humor — but comes with complexity and reward hacking risk
- **DPO** achieves similar goals with a simpler pipeline — increasingly the default choice
- **Personality** can be injected via system prompts (fast, shallow), fine-tuning on persona data (deeper), or LoRA adapters (modular, production-friendly)
- **Evaluation** needs both human judgment and synthetic scoring — the Big Five framework gives you a structured way to probe personality
- **Scalable personalization** comes from user modeling (embeddings, memory injection, RAG) and synthetic data generation

The stack is: SFT for structure → RLHF/DPO for personality → LoRA for modularity → user modeling for personalization. Each layer adds depth.

Happy post-training!
