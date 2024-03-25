# Week2-1 (Fine-tuning LLMs with instruction)
## Introduction
1. Instruction fine-tuning
2. Catastrophic forgetting
3. PEFT
4. LoRA

## Instruction fine-tuning
1. Limitations of in-context learning
   - In-context learning may not work for smaller models, even with multiple examples
   - Examples take up space in the context window
   - Instead, try FINE-TUNING the model
2. Fine-tuning
   - In contrast to pre-training, where you train the LLM using vast amounts of unstructured textual data via selfsupervised learning, fine-tuning is a supervised learning process where you use a dataset of labeled examples to update the weights of LLM.
   - The labeled examples are prompt-completion pairs
   - The fine-tuning process extends the training of the model to improve its ability to generate good completions for a specific task
3. Instruction fine-tuning
   - particularly good at improving a model's performance on a variety of tasks.
   - Instruction fine-tuning trains the model using examples that demonstrate how it should respond to a specific instruction.
   - The dataset you uuse for training includes many pairs of prompt-completion examples for the task you're interested in, each of which includes an instruction.
   - These prompt-completion examples allow the model to learn to generate responses that follow the given instructions.
  ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/db72e51879722b5fbace8dee09ccbd24b44425cb/week2_screenshots/0014.jpg)
  ![image(https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/db72e51879722b5fbace8dee09ccbd24b44425cb/week2_screenshots/0015.jpg)
4. Instruction fine-tuning, where all of the model's weights are updated is known as full fine-tuning. Just like pre-training, full fine-tuning requires enough memory and compute budget to store and process all the gradients, optimizers and other components.
   Therefore, you can benefit from the parallel computing strategies.
5. The process of Instruction fine-tuning
   - 
