# Introduction to Generative AI

## Learning Objectives

## What is Generative AI?

1. [Welcome](#welcome)
2. [How Generative AI works](#how-generative-ai-works)
3. [LLMs as a thought partner](#llms-as-a-thought-partner)
4. [I is a general purpose technology](#ai-is-a-general-purpose-technology)
5. [Generative AI applications](#generative-ai-applications)

### Welcome

- What is generative AI?
  - Artificial intelligence systems that can produce high quality content, specifically text, images, and audio.
- Generative AI is also a developer tool
  - Shows the usage of langchain.

- What you'll learn
  - Week 1
    - How generative AI technology works
      - What it can and can't do
      - Common use cases
  - Week 2
    - Generative AI Projects
      - Identify and build Generative AI use cases
      - Technology options
  - Week 3
    - Impact on business and society
      - How teams can take advantage of Generative AI

### How Generative AI works

- Supervised learning (labeling things)
  - Generative AI is build using supervised learning.

- Generating text using Large Language Models (LLMs)

### LLMs as a thought partner

### AI is a general purpose technology

- Examples of tasks LLMs can carry out
  - Writing
  - Reading
  - Chatting

- Web-based vs software application use of LLMs

## Generative AI applications

### Writing

### Reading

### Chatting

### What LLMs can and cannot do

- Limitations
  - Knowledge cutoffs
    - LLM has knowledge upto a certain timestamp.
  - Hallucinations
  - The input (and output) length is limited
    - The total amount of context you can give is limited.
  - Generative AI does not work well with structured (tabular) data.
  - Bias and Toxicity

### Tips for prompting

- Be detailed and specific
  - Give sufficient context for LLM to complete the task.
  - Describe the desired task in detail.
- Guide the model to think through its answer
- Experiment and iterate

### Image generation

- Image generation (diffusion model)
  - Typically ~100 steps for diffusion model.
- Additional resources (KA)
  - [Intuitive explanation of Diffusion models from AssemblyAI](https://www.assemblyai.com/blog/how-physics-advanced-generative-ai/#generative-ai-with-thermodynamics)
    - Thermodynamics can be viewed as the study of randomness.
    - The likelihood of 50 % of N coins being heads-up compared to 100 % of N coins being heads-up increases (?) exponentially with N.
    - ?? Intuition behind considering food coloring concentrated in a single drop as 100 % heads-up whereas spread out evenly as 50 % heads-up.
    - This process is called *diffusion*.
      - Random motion of food coloring leads to uniform color.
    - Thermodaynamics: Views atoms as coins.
    - Diffusion models: Views pixels of images as atoms.
    - "Random motion" of pixels always leads to "TV static": image equivalent of uniform food coloring.
    - Different "drops" for Diffusion models correspond to different **types** of images.
    - Flow diagram describes the reasoning for using diffusion in forward time and diffusion model in reverse time to sample image.
      - Sampling from the training data distribution directly is hard.
    - Related Physics concept: [Poisson Flow Generative Models](https://www.assemblyai.com/blog/an-introduction-to-poisson-flow-generative-models/)
  
  - [Diffusion Models - Introduction](https://www.assemblyai.com/blog/diffusion-models-for-machine-learning-introduction/)
    - Advantages over GANs
      - Does not require adversarial training
      - Scalability and parallilizability
    - [Isotropic Gaussian distribution](https://math.stackexchange.com/questions/1991961/gaussian-distribution-is-isotropic)

### Quiz

## Resources

### Web UI chatbots

- [ChatGPT](https://chat.openai.com/)
- [Bard](https://bard.google.com/chat)
- [Bing Chat](https://www.bing.com/search?q=Bing+AI&showconv=1&FORM=hpcodx)
