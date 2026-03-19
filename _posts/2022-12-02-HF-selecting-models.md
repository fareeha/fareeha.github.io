---
title: "Hugging Faces - selecting models"
date: 2022-12-02
categories: [Machine Learning]
tags: [hugging_face, opensource models]
---


A practical walkthrough of how to find, evaluate, and use open-source models from the Hugging Face ecosystem — covering everything from model selection to running NLP, audio, and vision tasks with the `transformers` library.

---

## Models Page

Finding the right model starts on the [Models page](https://huggingface.co/models):

- **Identify your task** — e.g., transcribe speech in French
- **Select a permissive license** (like Apache 2.0) — this allows commercial use
- **Sort by downloads or trending** to find battle-tested models
- **Check the model card** — look for architecture details, training info, and available checkpoints (which often come in varying sizes)

### How to Estimate Memory Needed

A quick trick to estimate how much memory you'll need to run a model:

- Go to **Files and Versions** on the model page
- Look for `pytorch_model.bin` (or the safetensors equivalent)
- Note the file size — e.g., if it says 3.09 GB, that's roughly how much memory you need
- **Multiply by 1.2** (add ~20% overhead) and that's your realistic memory requirement

## Alt: Task Page

Another way to discover models:

- Select a task from the [Tasks page](https://huggingface.co/tasks)
- Browse models specifically built for that task, along with relevant datasets and demos

---

## Once a Model is Selected

Use the `transformers` library to import and run the model. You have two main options:

- **Pipeline** — a high-level helper that handles everything for you
- **AutoProcessor / AutoModel** — load components directly for more control

The `Pipeline` object is great because it takes care of all the preprocessing your inputs need to match the model's expectations. For example, some audio models expect input as a log-mel spectrogram, text needs to be converted to input tokens, and images need to be resized and normalized. With Pipeline, you don't have to do any of this by hand.

```python
from transformers import pipeline

# Load a conversational model using the high-level pipeline API
chatbot = pipeline(task="conversational",
                   model="./models/facebook/blenderbot-400M-distill")

user_message = """
What are some fun activities I can do in the winter?
"""

from transformers import Conversation

# Wrap the message in a Conversation object to track multi-turn dialogue
conversation = Conversation(user_message)
print(conversation)

# Continue the conversation by adding follow-up messages
conversation.add_message(
    {"role": "user",
     "content": """
What else do you recommend?
"""
    })
```

If the answer isn't great, try adding previous prompts as context — the model benefits from seeing the full conversation history.

## Open LLM Leaderboard

Worth bookmarking: the [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard) ranks open-source models across multiple benchmarks. Great for comparing before you commit to a model.

---

## Translation and Summarization

### Translation

```python
# Load a multilingual translation model in bfloat16 to cut memory usage in half
translator = pipeline(task="translation",
                      model="./models/facebook/nllb-200-distilled-600M",
                      torch_dtype=torch.bfloat16)
```

**Why bfloat16?** Using `bfloat16` instead of `float32` (the default) cuts memory requirements roughly in half:

- `float32` uses 32 bits (4 bytes) per parameter
- `bfloat16` uses 16 bits (2 bytes) per parameter

So a model that's X GB in float32 becomes ~X/2 GB in bfloat16. You also get faster inference (often 1.5–2x speedup), lower memory bandwidth requirements, and faster matrix multiplications — especially on hardware with native bfloat16 support like NVIDIA A100s and TPUs.

```python
# Translate English to French using language codes
text_translated = translator(text,
                             src_lang="eng_Latn",
                             tgt_lang="fra_Latn")
```

When you're done with a model, free up memory explicitly:

```python
import gc

# Delete the model and trigger garbage collection to reclaim GPU/CPU memory
del translator
gc.collect()
# Returns the number of unreachable objects freed (model, tensors, caches, buffers)
```

### Summarization

```python
# Load a summarization model — BART-large-CNN is a solid default
summarizer = pipeline(task="summarization",
                      model="./models/facebook/bart-large-cnn",
                      torch_dtype=torch.bfloat16)

# min_length and max_length control the bounds of the generated summary
summary = summarizer(text,
                     min_length=10,
                     max_length=100)
```

---

## Sentence Embeddings

Sentence embeddings let you measure how close two sentences are in meaning. Useful for information retrieval, clustering, and semantic search.

```python
from sentence_transformers import SentenceTransformer

# MiniLM is a lightweight model that produces 384-dim embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

sentences1 = ['The cat sits outside',
              'A man is playing guitar',
              'The movies are awesome']

# Encode sentences into dense vectors — convert_to_tensor for GPU-friendly ops
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
# Output shape: torch.Size([3, 384]) — one 384-dim vector per sentence

from sentence_transformers import util

# Compute pairwise cosine similarity between two sets of sentence embeddings
cosine_scores = util.cos_sim(embeddings1, embeddings2)
# Higher score = more semantically similar
# e.g., 0.65 between "The movies are awesome" and "The films are great"
```

---

## Zero-Shot Audio Classification

This is great for tasks like identifying what language is being spoken or what birds are in an area. Typically, classification requires a model trained on the exact classes you want to identify — but with zero-shot, you just pass in candidate labels and the model picks the most plausible match.

```python
from IPython.display import Audio as IPythonAudio

# Play the audio sample in a notebook — the rate must match the sample's sampling rate
IPythonAudio(audio_sample["audio"]["array"],
             rate=audio_sample["audio"]["sampling_rate"])

# Load a zero-shot audio classifier — no fine-tuning needed for new classes
zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="./models/laion/clap-htsat-unfused")
```

**A note on sampling rates:** Sound is a continuous signal. To get a digital representation, we sample the analog waveform at regular intervals. The **sampling rate** (in Hz) is the number of samples taken per second — 48,000 Hz and 44,100 Hz are both common and generally interchangeable for most models.

```python
# Define the classes you want to distinguish — the model finds the best match
candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]

zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)
# [{'score': 0.998, 'label': 'Sound of a dog'},
#  {'score': 0.001, 'label': 'Sound of vacuum cleaner'}]
```

The model doesn't "know" these classes — it just scores which label is most plausible given the audio.

---

## Automatic Speech Recognition

Think meeting notes, dictation, transcription — converting audio to text.

```python
# Grab a single example from the dataset
example = next(iter(dataset))

# Distil-Whisper is a smaller, faster version of OpenAI's Whisper model
asr = pipeline(task="automatic-speech-recognition",
               model="distil-whisper/distil-small.en")

# Pass raw audio array directly — the pipeline handles resampling and preprocessing
asr(example["audio"]["array"])
```

---

## Text to Speech

Text-to-speech is a one-to-many problem (many valid ways to say the same sentence), which makes it inherently more challenging than speech-to-text.

```python
# VITS is a lightweight end-to-end TTS model
narrator = pipeline("text-to-speech",
                    model="./models/kakao-enterprise/vits-ljs")

narrated_text = narrator(text)

from IPython.display import Audio as IPythonAudio

# Play the generated audio — sampling_rate must match what the model produced
IPythonAudio(narrated_text["audio"][0],
             rate=narrated_text["sampling_rate"])
```

---

## Object Detection

Object detection combines classification (what is it?) and localization (where is it?) in a single pass. A practical use case: helping a visually impaired person understand what's in a picture.

```python
# DETR (DEtection TRansformer) — a transformer-based object detector
od_pipe = pipeline("object-detection", "./models/facebook/detr-resnet-50")

raw_image = Image.open('huggingface_friends.jpg')
raw_image.resize((569, 491))

# Returns bounding boxes, labels, and confidence scores for each detected object
pipeline_output = od_pipe(raw_image)

# Helper to draw bounding boxes on the image
processed_image = render_results_in_image(raw_image, pipeline_output)

# Convert detections into a natural language description
text = summarize_predictions_natural_language(pipeline_output)
```

You can even chain this with TTS for an audio narrative of the image:

```python
# Pipe the object detection summary into text-to-speech
tts_pipe = pipeline("text-to-speech",
                    model="./models/kakao-enterprise/vits-ljs")
narrated_text = tts_pipe(text)

IPythonAudio(narrated_text["audio"][0],
             rate=narrated_text["sampling_rate"])
```

---

## Image Segmentation

Segmentation goes beyond bounding boxes — it identifies the exact pixels that belong to an object. With visual prompting, you can point at something in a picture and the model segments just that object.

```python
# SlimSAM — a lightweight version of Meta's Segment Anything Model
sam_pipe = pipeline("mask-generation",
    "./models/Zigeng/SlimSAM-uniform-77")

# Generate masks for the entire image — higher points_per_batch = faster inference
output = sam_pipe(raw_image, points_per_batch=32)

# Or provide a specific point to segment a particular object
# Input can be a 2D point coordinate or a bounding box
input_points = [[[1600, 700]]]
inputs = processor(raw_image,
                 input_points=input_points,
                 return_tensors="pt")

# Run inference without tracking gradients (faster, less memory)
with torch.no_grad():
    outputs = model(**inputs)

# Post-process the raw mask predictions back to original image dimensions
predicted_masks = processor.image_processor.post_process_masks(
    outputs.pred_masks,
    inputs["original_sizes"],
    inputs["reshaped_input_sizes"]
)
```

---

## Image Retrieval

Image retrieval uses multimodal embeddings to match images with text descriptions (or other images). The idea: encode both images and text into the same embedding space, then score how well they align.

```python
from transformers import CLIPModel, CLIPProcessor

# CLIP jointly embeds images and text — trained on 400M image-text pairs
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Encode an image and candidate text descriptions together
inputs = processor(
    text=["a photo of a woman and a dog", "a photo of a sunset", "a photo of a car"],
    images=raw_image,
    return_tensors="pt",
    padding=True
)

outputs = model(**inputs)

# Higher score = better match between image and text
logits_per_image = outputs.logits_per_image
probs = logits_per_image.softmax(dim=1)
# e.g., [0.92, 0.05, 0.03] — the first description is the best match
```

"Multimodal" just means working with more than one type of data — here, images and text together.

---

## Image Captioning

Image captioning generates a natural language description of an image. The model looks at the image and produces a sentence (or paragraph) describing what it sees.

```python
from transformers import pipeline

# BLIP is a strong general-purpose image captioning model
captioner = pipeline("image-to-text",
                     model="Salesforce/blip-image-captioning-base")

# Pass in an image and get back a text description
caption = captioner(raw_image)
# [{'generated_text': 'a group of people standing next to each other'}]
```

You can also do **conditional captioning** — give the model a text prompt to steer the description:

```python
# Start the caption with a prompt to guide the output
caption = captioner(raw_image, text="a photo of")
# [{'generated_text': 'a photo of a group of friends at a park'}]
```

---

## Multimodal Visual Question Answering (VQA)

VQA lets you ask natural language questions about an image and get an answer back. It combines vision and language understanding in a single model.

```python
from transformers import pipeline

# BLIP also supports VQA — same model family, different task head
vqa = pipeline("visual-question-answering",
               model="Salesforce/blip-vqa-base")

# Ask a question about the image
answer = vqa(image=raw_image,
             question="How many people are in this photo?")
# [{'score': 0.95, 'answer': '4'}]

# You can ask follow-up questions about the same image
answer = vqa(image=raw_image,
             question="What are they wearing?")
```

---

## Zero-Shot Image Classification

Just like zero-shot audio classification, this lets you classify images into categories the model was never explicitly trained on. You provide candidate labels and the model scores which one fits best.

```python
from transformers import pipeline

# CLIP-based zero-shot classifier — no fine-tuning needed for new classes
classifier = pipeline("zero-shot-image-classification",
                      model="openai/clip-vit-base-patch32")

# Define whatever classes you want — the model will rank them
candidate_labels = ["a photo of a cat",
                    "a photo of a dog",
                    "a photo of a bird"]

result = classifier(raw_image, candidate_labels=candidate_labels)
# [{'score': 0.89, 'label': 'a photo of a dog'},
#  {'score': 0.08, 'label': 'a photo of a cat'},
#  {'score': 0.03, 'label': 'a photo of a bird'}]
```

The beauty of zero-shot: you can change your classes at inference time without retraining anything.

---

## Deployment

Once you've picked and tested a model, here are the main paths to get it into production:

**Hugging Face Inference Endpoints** — the simplest option. Deploy any model from the Hub as a dedicated API endpoint with a few clicks. You choose the hardware (CPU or GPU), and HF handles scaling, hosting, and infrastructure.

**Hugging Face Inference API** — a serverless option for quick prototyping. Free tier available, but rate-limited. Good for testing before committing to dedicated infrastructure.

**Export + Self-Host** — for full control, export your model to ONNX or TorchScript and serve it with frameworks like FastAPI, TorchServe, or Triton Inference Server. More setup, but you own the entire stack.

**Optimizations to consider before deploying:**

- **Quantization** — reduce model precision (float32 → int8 or int4) for faster inference and lower memory
- **Distillation** — train a smaller model to mimic the larger one (e.g., DistilBERT, Distil-Whisper)
- **Batching** — group multiple requests together for better GPU utilization
- **Caching** — cache frequent predictions to avoid redundant inference

The right deployment choice depends on your scale, latency requirements, and budget. For most projects, start with Inference Endpoints and optimize from there.

---

## Recap

- **Start at the Models page** — filter by task, license, and popularity
- **Estimate memory** from the model file size (×1.2 for overhead)
- **Use Pipeline** for fast prototyping — it handles all the preprocessing
- **bfloat16** is a free win for cutting memory and speeding up inference
- **Zero-shot models** let you skip fine-tuning entirely for classification tasks
- **Chain models together** for powerful workflows (e.g., object detection → TTS)
- **Deploy** with Inference Endpoints for simplicity, or self-host for control

Happy model hunting! 
