---
title: "Creating a 🧪 Longevity & Biohacking agentic RAG using LlamaIndex"
date: 2021-06-19
categories: [Machine Learning, Tools]
tags: [rag, llamaindex, agent, python, ml-demos]
---


LLMs have leveled up my *self-care* game, so I played around with building a longevity personal research assistant to summarize findings and answer questions. Here's how you can built one too:

I used 🦙 **LlamaIndex** which is a framework designed to build intelligent agents skilled in reasoning and decision-making with your own data. 

> **Why LlamaIndex?**
LangChain and LlamaIndex are both popular frameworks for working with LLMs. LangChain is great at connecting different tools and steps in an AI workflow, while LlamaIndex excels at efficient data indexing and retrieval, making it ideal for RAG pipelines.


In this tutorial, we'll start with the **building blocks** of LlamaIndex and gradually move toward building a sophisticated **Longevity Agentic RAG** system.

Jump right to the code *here*.

## 🧪 Longevity & Biohacking Assistant

**Task:** Explore papers and protocols related to longevity supplements and regimens from specific sources. We'll perform one-shot query (ask question, request summary) and a multi-step reasoning query (a more complex query requiring reasoning).
* QA example: *"Retrieve research findings for anti-aging benefits of supplements like urolithin A or resveratrol"*
* QA example: *"What are the most effective evidence-based biohacks for extending healthspan?"*
* QA example: *"How do specific supplements like NMN or resveratrol influence aging processes?"*
* QA example: *"What role does caloric restriction play in longevity, and what are its underlying mechanisms?"*
* QA example: *"How do lifestyle practices such as sleep quality, stress management, and social connections impact aging?"*
* QA example: *"What insights can we gain from centenarian studies, like those conducted in Okinawa, about healthy aging?"*
* Summary example: *"Summarize protocols from biohackers or doctors"* (TODO)
* Agent Reasoning Loop example: *“What’s a good pre-breakfast morning supplement stack for energy and skin health?”*


📄 Documents Used:
1.	Extending Human Health Span and Longevity—A Symposium Report
    [PMC Article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10231756/pdf/nihms-1732430.pdf)
2.	Bio-Hacking Better Health—Leveraging Metabolic Biochemistry to Extend Healthspan
[MDPI Article](https://www.mdpi.com/2076-3921/12/9/1749)



---

## 🧱 Building Blocks

Our first step is to create *indexes* from the documents. Next, we’ll build *query engines* around those indexes. When a query comes through, a custom *selector* will route it to the appropriate query engine. It'll make more sense once we start building.

### Step 1. Convert Documents to Nodes
First, we split our documents into smaller units called **nodes** of size `1024`

```python
from llama_index.core.node_parser import SentenceSplitter

splitter = SentenceSplitter(chunk_size=1024)
nodes = splitter.get_nodes_from_documents(documents)
```

### Step 2. Indexing Nodes
Create different types of indexes based on the desired retrieval needs. For our case we need indexes for **QA** and **Summary** generation as follows:

- **Vector Index**: Uses text embeddings for similarity-based retrieval
- **Summary Index**: Retrieves all content in the index regardless of query

```python
from llama_index.core import SummaryIndex, VectorStoreIndex

summary_index = SummaryIndex(nodes)
vector_index = VectorStoreIndex(nodes)
```

### Step 3. Configure the LLM and Embeddings

Select your desired LLM and text-embedding model. *Quality of retrievals will depend on model selected for use-case and context.*

```python
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
```

### Step 4. Create Query Engines

Engines create the desired response behaviours for the indexes.

```python
summary_query_engine = summary_index.as_query_engine(
    response_mode="tree_summarize",
    use_async=True,
)
vector_query_engine = vector_index.as_query_engine()
```


### Step 5. From Query Engine to Tool

We can use `QueryEngineTool` to wrap each query engine with a helpful description.

```python
from llama_index.core.tools import QueryEngineTool

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description="Useful for summarization questions related to MetaGPT"
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description="Useful for retrieving specific context from the MetaGPT paper."
)
```


### Building a Router

As the name suggests, the **Router** is responsible for selecting the appropriate **EngineTool** we created earlier. It essentially routes each query to the most relevant engine based on its content.


### LlamaIndex has these two Selector types to build a Router:
- **LLM Selector**: Uses reasoning via LLM to choose a tool.
- **Pedantic Selector**: Uses keyword rules to pick a tool (faster but rigid).

For our case, we'll use the `LLMSingleSelector`:

```python
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector

query_engine = RouterQueryEngine(
    selector=LLMSingleSelector.from_defaults(),
    query_engine_tools=[summary_tool, vector_tool],
    verbose=True
)
```


```python
response = query_engine.query("Give me a summary of MetaGPT") # Will invoke `summary_tool`
print(len(response.source_nodes))  # Should be equal to total chunks i.e. summary over entire document

response = query_engine.query("What are the high-level results of MetaGPT?")  # Will invoke `vector_tool`
print(len(response.source_nodes))  # Display k chunks, i.e. top-k matching vectors returned
```

---

## 🛠️ Inferring Function Calls and Parameters with an LLM

Now let’s explore a powerful concept: using an LLM to infer which function to call and automatically populate its parameters based on a natural language query.

Here’s a simple example using `predict_and_call` to demonstrate how this works:

```python
from llama_index.core.tools import FunctionTool

def add(x: int, y: int) -> int:
    return x + y

def multiply(x: int, y: int) -> int:
    return x * y

add_tool = FunctionTool.from_defaults(fn=add)
multiply_tool = FunctionTool.from_defaults(fn=multiply)

response = Settings.llm.predict_and_call(
    [add_tool, multiply_tool], 
    "Tell me the output of the multiply function on 2 and 9",
    verbose=True
)
print(response) # 18
```
 
Similarly, when querying over documents, we can use `MetadataFilters` to gain fine-grained control over which parts of the documents to query e.g, filtering by page number. TODO : insert an actual query - specific to some page eg page 2 

```python
from llama_index.core.vector_stores import MetadataFilters

query_engine = vector_index.as_query_engine(
    similarity_top_k=2,
    filters=MetadataFilters.from_dicts([{"key": "page_label", "value": "2"}])
)
```

---

## 🤖 Agent with Reasoning Loop

So far we;ve build a single pass query engine : given a query, find the right tool and return the response. 

But what if we want to ask complex queries that require multiple reasoning steps? For example:

*"Compare how the two papers define and approach biohacking. What strategies do they both endorse, and where do they differ?"* — the agent must retrieve summaries from both papers, compare techniques, and infer differences.

*"Based on the discussed strategies in both documents, which interventions are the most accessible for the average person, and which require clinical oversight?"* — the agent needs to parse descriptions of each intervention, then apply reasoning about accessibility and clinical use.

*"Can you create a daily routine based on evidence-based biohacks from both papers that supports NAD+ levels and reduces inflammation?"* — the agent combines information from NAD+ sections and anti-inflammatory practices, then arranges them chronologically.

*"Which biohacking strategies mentioned in the symposium report are supported by metabolic mechanisms discussed in the MDPI paper?"* — this requires cross-document linking, where the agent identifies common themes and overlaps.


Here we'll build an agent that reasons over tools in multiple steps. We'll use `FunctionCallingAgent` implementation which is an **Agent** that natively integrates with the Function Calling capabilities of LLMs.

In *LlamaIndex*, an **Agent** is composed of two key components:

- AgentWorker: Executes individual reasoning steps. It takes the current state of the task and determines the next function/tool to call along with its arguments.

- AgentRunner: Manages the full task execution loop. It initializes the task, orchestrates multiple runs of the `AgentWorker`, and aggregates the intermediate results into a final response.

This modular architecture allows for iterative tool usage, dynamic decision-making, and more complex task completion.


### Step 1: Create the AgentWorker

We pass the tools we created above to the `AgentWorker` and an LLM

```python
from llama_index.agent import FunctionCallingAgentWorker

agent_worker = FunctionCallingAgentWorker.from_tools(
    [vector_tool, summary_tool],
    llm=Settings.llm,
    verbose=True
)
```

### Step 2: Create the AgentRunner

We pass the AgentWorker to the `AgentRunner` to create our Agent.

```python
from llama_index.agent import AgentRunner

agent = AgentRunner(agent_worker)
response = agent.query("Tell me about the agent roles in MetaGPT and how they communicate")
```
Lets look at the verbose output :

```
Added user message to memory: Tell me about the agent roles in MetaGPT, and then how they communicate with each other.

=== Calling Function ===
Calling function: summary_tool_metagpt with args: {"input": "agent roles in MetaGPT"}
```

Notice how the agent breaks down the question into steps—identifying the query about "agent roles in MetaGPT" and calling `summary_tool_metagpt` to answer it.

While this is a reasonable choice, a more precise approach might involve using the `vector_tool`, which could retrieve more relevant context for this specific question. A more capable model, such as GPT-4 Turbo, Claude 3 Sonnet, or Opus, might have selected the `vector_tool` instead for improved accuracy. 

Regardless we're able to get the correct output :

```
=== Function Output ===
The agent roles in MetaGPT include Product Manager, Architect, Project Manager, Engineer, and QA Engineer. Each role has specific responsibilities tailored to different aspects of the collaborative framework, such as conducting business-oriented analysis, translating requirements into system design components, handling task distribution, executing code, and formulating test cases to ensure code quality. These roles work together in a sequential workflow following Standard Operating Procedures (SOPs) to efficiently tackle complex tasks within the MetaGPT framework.
```

It uses this output to perform Chain of Thought (CoT) to then trigger the next question:

```
=== Calling Function ===
Calling function: summary_tool_metagpt with args: {"input": "how agents communicate with each other in MetaGPT"}

=== Function Output ===
Agents in MetaGPT communicate with each other through structured communication interfaces, message pools, and a publish-subscribe mechanism. They utilize a shared message pool to publish structured messages and can subscribe to relevant messages based on their profiles. This approach enhances role communication efficiency within the framework, allowing agents to obtain directional information from other roles and public information from the environment. Additionally, they communicate through mechanisms such as message pools and subscriptions, enabling efficient sharing within the simulated software company environment.
```
Now it combines this entire conversation hostory to geenrate the final response to "Tell me about the agent roles in MetaGPT and how they communicate":

```
=== LLM Response ===
In MetaGPT, the agent roles include Product Manager, Architect, Project Manager, Engineer, and QA Engineer. These roles have specific responsibilities tailored to different aspects of the collaborative framework. They communicate with each other through structured communication interfaces, message pools, and a publish-subscribe mechanism. This approach enhances role communication efficiency within the framework, allowing agents to obtain directional information from other roles and public information from the environment.
```

In LlamaIndex, an agent powered by an LLM typically performs the following steps:

- **Decomposes the input query** into smaller subtasks (if needed)
- **Selects and calls tools step-by-step** to handle each subtask (via function calling)
- **Maintains memory buffer**, including chat history and intermediate tool outputs
- **Generates a final response using the LLM**, informed by the stored memory and tool outputs

This loop continues until the task is complete, enabling complex multi-step reasoning across tools and contexts.

### Debugging Tools

You can also step through the agent's reasoning manually, which is super helpful for debugging: 

```python
task = agent.create_task(
    "Tell me about the agent roles in MetaGPT, "
    "and then how they communicate with each other."
)

step_output = agent.run_step(task.task_id)

completed_steps = agent.get_completed_steps(task.task_id)
print(f"Num completed for task {task.task_id}: {len(completed_steps)}")
print(completed_steps[0].output.sources[0].raw_output)

upcoming_steps = agent.get_upcoming_steps(task.task_id)
print(f"Num upcoming steps for task {task.task_id}: {len(upcoming_steps)}")
upcoming_steps[0]

step_output = agent.run_step(
    task.task_id, input="What about how agents share information?"
)

step_output = agent.run_step(task.task_id)
print(step_output.is_last)

response = agent.finalize_response(task.task_id)
print(str(response))
```

---

## 📚 Extend to handle multiple Documents

If you have many papers, each gets its own summary and vector tool. We collect them all into a list:

```python
paper_to_tools_dict = {}
for paper in papers:
    vector_tool, summary_tool = get_doc_tools(paper, Path(paper).stem)
    paper_to_tools_dict[paper] = [vector_tool, summary_tool]

initial_tools = [tool for tools in paper_to_tools_dict.values() for tool in tools]
```

With 2 papers, we now have 4 tools (2 per paper). Now we can ask the agent questions that span across both documents:

```python
response = agent.query(
    "Compare how the two papers define and approach biohacking. "
    "What strategies do they both endorse, and where do they differ?"
)
```
TODO: insert output 

### Tool Explosion Problem

This works great for a handful of papers. But what happens when you have 100 or even 1,000 documents? Things start to break down:

More papers == More tools 
- Context window issues - multiple tools per document may not all fit in the prompt 
- Cost & latency spike - more tools means more tokens per request
- Tool confusion - the LLM struggles to choose correctly when the list of tools is too large

We need a more sophisticated solution - enter *RAG*.

### 🔎 Solution: Tool RAG (retrieve tools, not chunks!)


The idea here is elegant: instead of doing retrieval at the text-chunk level, we do retrieval at the tool level. When a query arrives, we first retrieve a small set of relevant tools, and only then pass those to the agent.

LlamaIndex agents let you plug in a tool retriever to accomplish this. Since our tools are stored as a list of Python objects in all_tools, we need a way to convert and serialize these objects into searchable representations. LlamaIndex provides the `ObjectIndex` abstraction for exactly this — it lets us define an Object Index and Object Retriever over our tools, so the retriever returns actual tool objects ready to use:

```python
from llama_index.core.objects import ObjectIndex

obj_index = ObjectIndex.from_objects(
    all_tools,
    index_cls=VectorStoreIndex,
)
obj_retriever = obj_index.as_retriever(similarity_top_k=3)

tools = obj_retriever.retrieve("Tell me about the eval dataset in MetaGPT and SWE-Bench")
```

You can also plug object retriever tools into a tool-aware agent:

```python
agent_worker = FunctionCallingAgentWorker.from_tools(
    tool_retriever=obj_retriever,
    llm=Settings.llm,
    system_prompt="""
You are an agent designed to answer queries over a set of given papers.
Please always use the tools provided to answer a question. Do not rely on prior knowledge.
""",
    verbose=True
)
agent = AgentRunner(agent_worker)
```

---

## Recap

- **Start simple**: Documents → Nodes → Index → Query
- **Add abstraction**: Wrap queries into tools
- **Add intelligence**: Use selectors or full agents
- **Scale**: Use RAG over tools to handle many documents

LlamaIndex makes it possible to build rich agentic systems with your own data, with flexibility and power under the hood.

Happy building! 🦙✨
