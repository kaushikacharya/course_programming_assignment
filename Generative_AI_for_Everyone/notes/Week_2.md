# Generative AI Projects

## Software applications

### Using Generative AI in software applications

### Trying Generative AI code yourself

- [Prompting an LLM in code](https://learn.deeplearning.ai/genai4e/lesson/1/activity1)
- [Reputation monitoring system](https://learn.deeplearning.ai/genai4e/lesson/2/activity2)

### Lifecycle of a Generative AI project

- Tools to improve performance
  - Prompting
  - Retrieval augmented generation (RAG)
    - Give LLM access to external data sources
  - Fine-tune models
    - Adapt LLM to your task
  - Pretrain models
    - Train LLM from scratch

- Example of RAG
  - Initially the food ordering system failed to answer how many calories are there in a recipe. Later it acquired knowledge and replied the calories quantity.

### Cost intuition

- Approximate cost computed based on the number of words read by an average reader for an hour.

### Quiz

## Advanced technologies: Beyond prompting

### Retrieval Augmented Generation (RAG)

- Examples of RAG applications
  - Chat with PDF files
    - pandachat
    - chatpdf to name a few.
  - Answer questions based on a website's articles
    - Examples
      - Coursera Coach
      - Snapchat
      - Hubspot
  - New form of web search
    - Examples
      - Microsoft/Bing chat
      - Google
      - You.com

- Big Idea: LLM as reasoning engine
  - Use LLM as a reasoning engine to process information, rather than using it as a source of information.

### Fine-tuning

- Why fine-tune?
  - To carry out a task that isn't easy to define in a prompt.
    - Examples:
      - Summarize in certain style or structure
      - Mimicking a writing or speaking style
  - To help LLM gain specific knowledge
    - Examples
      - Medical notes
      - Legal documents
      - Financial documents
  - To get a smaller model to perform a task
    - Lower cost/latency to deploy
    - Can run on mobile/laptop (edge devices)

### Pretraining an LLM

- When should you pretrain an LLM?
  - For building a specific application:
    - Option of last resort
    - Could help if have a highly specialized domain

### Choosing a model

- Criteria
  - Model size
  - Open source or closed source

### How LLMs follow instructions: Instruction tuning and RLHF

- Instruction tuning
  - Fine tune to answer in a next word prediction format.

- Reinforcement learning from human feedback (RLHF)
  - This technique improves the quality of answer better than instruction tuning.
  - Objective:
    - Helpful, Honest, Harmless
  - Step 1: Train an answer quality (reward) model
  - Step 2: Have LLM generate a lot of answers. Further train it to generate more responses that get high scores.

### Tool use and agents

- Tools for reasoning
  - LLMs are not good at precise math.
  - Example shown to use external calculator to compute compound interest.
- Agents
  - Use LLM to choose and carry out complex sequences of actions.
