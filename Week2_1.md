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
  
## Multi-task instruction fine-tuning
1. For multi-task fine-tuning, the training dataset is comprised of example inputs and outputs for multiple tasks.
   - By training the model on the mixed dataset so that it can improve the performance of the model on all the tasks simultaneously thus avoiding the issue of catastrophic forgetting.
2. One drawback to multitask fine-tuning is that it requires a lot of data (50~100,000 examples in your training set)
3. Instruction fine-tuning with FLAN (Fine-tuned language net)
   - a specific set of intstructins used to fine-tune different models
   - 'The metaphorical dessert to the main course of pre-training'
4. FLAN-T5
   - the FLAN instruct version of the T5 foundation model
   - a great, general purpose, intstruct model
   - FLAN-T5 has been fine-tuned on 473 datasets across 146 task categories. Those datasets are chosen from other models and papaers as shown in the photo.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/e524d994d7c043331b413f0b3c47a481192de750/week2_screenshots/0036.jpg)
5. FLAN-PALM
   - the flattening struct version of the palm foundation model
6. SAMSum is a dataset  with messenges like conversations with summaries
   - The dialogues and summaries were crafted by linguists.
   - The prompt templeate is actually comprised of several different instructions that all basically ask the model to do the same thing, summarize the dialogue.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/46c0abbd89fbd9a0bace5a422e5bac5734e17418/week2_screenshots/0039.jpg)
7. Including different ways of saying the same instruction helps the model generalize and perform better.

## Scaling instruct models
1.  The paper 'Scaling Instruction-Finetuned Language Models' demonstrates that by fine-tuning the 540B PaLM model on 1836 tasks while incorporating Chain-of-Thought Reasoning data, FLAN achieves improvements in generalization, human usability, and zero-shot reasoning over the base model. The paper also provides detailed information on how each these aspects was evaluated.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d525cf640531634cd04a0a418f0e879a842b19b1/week2_screenshots/FLAN.png)

## Model Evaluation
1. LLM Evaluation - Challenges
   - In traditional machine learning, you can assess how well a model is doing by looking at its performance on training and validation datasets where the output is already known. The models are deterministic. Accuracy = Correct Predictions / Total Predictions
   - For LLM where the output is non-deterministic and language-based evaluation is much more challenging.
2. ROUGE and BLEU are two widely used evaluation metrics for different tasks
3. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
   - 


