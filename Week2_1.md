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
  ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/db72e51879722b5fbace8dee09ccbd24b44425cb/week2_screenshots/0015.jpg)
4. Instruction fine-tuning, where all of the model's weights are updated is known as full fine-tuning. Just like pre-training, full fine-tuning requires enough memory and compute budget to store and process all the gradients, optimizers and other components.
   Therefore, you can benefit from the parallel computing strategies.
5. The process of Instruction fine-tuning
   - prepare your instruction dataset
   - training split (training, validation, test)
   - pass the prompt into LLM
   - compare the complement with the label -> loss: cross-entropy
   - update the model weights
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/fda9cce43fb43897e5b1b266c16351572c7fca30/week2_screenshots/0018.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/fda9cce43fb43897e5b1b266c16351572c7fca30/week2_screenshots/0020.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/fda9cce43fb43897e5b1b266c16351572c7fca30/week2_screenshots/0021.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/fda9cce43fb43897e5b1b266c16351572c7fca30/week2_screenshots/0022.jpg)
6. Sample prompt instruction templates
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a9736fcc80e7d680ef963bdfd734cbd7cbd19e3b/week2_screenshots/0017.jpg)

## Fine-tuning on a single task
1. Somtimes, you can fine-tune a pre-trained model to improve performance on only the task that is of interest to you.
   - Good results can be achieved with relatively few examples.(Often 500~1000 examples)
2. There is a potential downside to fine-tuning on a single task -> 'catastrophic forgetting'
   - why? because the full fine-tuning process modifies the weights of the original LLM. While this leads to great performance on single fine-tuning task, but degrade performance on other tasks.
3. What choices do we have to avoid catastrophic forgetting?
   - Decide wheter catastrophic forgetting actually impacts your use case.
     * reliable performance on single task: it's okay to have catastrophic forgetting
     * want to maintain its multitask generalized capabilities: you can perform fine-tuning on multiple tasks at one time. (Good multitask fine-tuning may require 50~100,000 examples across many tasks, so will require more data and compute to train)
   - Parameter effiecient fine-tuning (PEFT)
4. PEFT
   - A set of techniques that preserves the weights of the original LLM and trains only a small number of task-specific adapter layers and parameters
   - PEFT show greater robustness to catastrophic forgetting since most of the pre-trained weighs are left unchanged.
  









