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
   - ROUGE is primarily employed to assess the quality of automatically generated summaries by comparing them to human-generated reference summaries.
4. BLEU (Bilingual Evaluation Understudy)
   - is an algorithm designed to evaluate the qulaity of machine-translated text by comparing it to human-generated translations.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/612fb4cbbe8106a33f15bc78d444d6e8946d2ca8/week2_screenshots/0053.jpg)
5. Some terminology
   - Unigram: single word
   - Bigram: two words
   - n-gram: a group of n-words
6. ROUGE-1
   - The recall metric measures the number of words or unigrams that are matched between the refernce and the generated output divided by the number of words or unigrams in the reference.
   - precision measures the unigram matches divided by the output size.
   - F1 score is the n=harmonic mean of both recall and precision.
   - Very basic metrics that only focused on individual words, and don't consider the ordering of the words. -> It can be deceptive. It's easily possible to generate sentences that score well bur would be subjectively poor.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/4fa1dd86ad50a603d4eaaee8ad695f2bb98b3546/week2_screenshots/0055.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/4fa1dd86ad50a603d4eaaee8ad695f2bb98b3546/week2_screenshots/0056.jpg)
   - You can get a slightly better score by taking into account bigrams at at time from the reference and generated sentence. -> you can consider the order of the words
   - By using bigrams, you can calculate ROUGE-2.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/056f5d4882baaae017999b77619a516e302b26ef/week2_screenshots/0058.jpg)
   - With longer sentences, they're greater chance that bigrams don't match.
   - You can look for the longest common subsequence present in both the generated output and the reference output.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/f326685ae806086cb3105715d7ed40ac31f36e1a/week2_screenshots/0059.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/f326685ae806086cb3105715d7ed40ac31f36e1a/week2_screenshots/0061.jpg)
   - You can only use the scores to compare the capabilities of models if the score were determined for the same task.
   - A particular problem with simple rouge scores is that it's possible for a bad completion to result in a good score. For example, in the next photo. One way you can counter the issue is by using a clipping function to limit the number of unigram matches to the maximum count for that unigram within the reference. However, you might still be challenged if their generated words are all present, but just in a different order. So, using a different rouge score can help experimenting with an n-gram size that will calculate the most useful score will be dependent on the sentence, the sentence size, and your use case.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/f326685ae806086cb3105715d7ed40ac31f36e1a/week2_screenshots/0063.jpg)
7. BLEU score
   - useful for evaluating the quality of machine-translated text
   - The score itself is calculated using the average precision over multiple n-gram sizes
   - quantifies the quality of a translation by checking how many n-grams in the machine-generated translation match those in the reference translation.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/f326685ae806086cb3105715d7ed40ac31f36e1a/week2_screenshots/0065.jpg)
8. Both rouge and BLEU are quite simple metrics and are relatively low-cost to calculate. Use them for simple reference as you iterate over your models, but you shouldn't use them alone to report the final evaluation. For overall evaluation of your model's performance, you will need to look at one of the evaluation benchmarks.

## Benchmarks 
1. Selecting the right evaluation dataset is important. And you should consider whether the model has seen your evaluation data during training.
2. Bechmarks, such as GLUE, SuperGLUE, or Helm, cover a wide range of tasks and scenarios.
3. GLUE (General Language Understanding Evaluation)
   - is a collection of natural language tasks (including sentiment analysis and question answering...)
4. SuperGLUE
   - a successor to GLUE
   - consists of a series of tasks such as multi-sentence reasoning, reading comprehension.
5. Both GLUE and SuperGLUE have leaderboards that can be used to compare and contrast evaluated models.
6. As models get larger, their performance on benchmarks start to match human ability on specific tasks. But there are not performing at human level at tasks in general. -> There is an arms racebetween the emergent properties of LLMs, and the benchmarks that aim to measure them.
7. MMLU (Massive Multitask Language Understanding) is designed for modern LLMs.
   - Models are tested on elementary mathematics, US history, computer science, law, and more -> tasks that extend way beyond basic language understanding
8. BIG-bench
   -consists of 204 tasks, ranging through linguistics, childhood development, math, common sense reasoning, biology, physics, social bias, software development and more.
   - BIG-bench comes in three different sizes, and part of the reason for this is to keep costs achievable, as running these large benchmarks can incur large inference costs.
9. HELM (Holistic Evaluation of Language Models)
    - Aims to improve the transparency of models, and to offer guidance on which models perform well for specific tasks.
    - HELM takes a multimetric approach, measuring seven metrics across 16 core scenarios, ensuring that trade-offs between models and metrics are clearly exposed.
    - One important feature of HELM is that it assesses on metrics beyond basic accuracy measures, like precision of the F1 score.
    - The benchmark also includes metrics for fairness, bias, and toxicity, which are becoming increasingly important to assess as LLMs become more capable of human-like language generation, and in turn of exhibiting potentially harmful behavior.
    - HELM is a living benchmark that aims to continuously evolve with the addition of new scenarios, metrics, and models

## Parameter efficient fine-tuning (PEFT)
1. Full fine-tuning requires memory not just to store the model, but various other parameters that are required during the training process.
2. PEFT method only update a small subset of parameters
   - Some PEFT techniques freeze most of the model weights and focus on fine tuning a subset of existing model parameters. Other techniques don't touch the original model weights at all, and instead add a small number of new parameters or layers and fine-tune only the new components.
   - So, the number of trained parameters is much smaller than the number of parameters in the original LLM (sometimes 15%~20% of the original LLM weights)
   - PEFT can often be performed on a single GPU
   - Since the original LLM is only slightly modified, PEFT is less prone to catastrophic forgetting problems like full fine-tuning.
3. Full fine-tuning results in a new version of the model for every task you train on. Each of these is the same size as the original model, so it can create an expensive storage problem if you're fine-tuning for multiple tasks.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/e21aee417db84221b4b3921a9f6ec3ee53196a0d/week2_screenshots/0084.jpg)
4. With parameter efficient fine-tuning, you train only a small number of weights, which results in a much smaller footprint overall. The new parameters are combined with the original LLM weights for inference. The PEFT weights are trained for each task and can be easily swapped out for inference, allowing efficient adaptation of the original model to multiple tasks.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/e21aee417db84221b4b3921a9f6ec3ee53196a0d/week2_screenshots/0085.jpg)
5. There are several methods you can use for parameter efficient fine-tuning, each with trade-offs on parameter efficiency, memory efficiency, training speed, model quality, and inference costs.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/e21aee417db84221b4b3921a9f6ec3ee53196a0d/week2_screenshots/0086.jpg)
6. Three main classes of PEFT methods
   - Selective methods
     * Fine-tune only a subset of the original LLM parameters
     * There are several approaches that you can take to identify which parameters you want to update. You can
train only certain components of the model or specific layers, or even individual parameter types. Researchers have found that the performance of these methods is mixed and there are significant trade-offs between parameter efficiency and compute efficiency. 
   - Reparameterization methods
     * also work with the original LLM parameters, but reduce the number of parameters to train by creating new low rank transformations of the original network weights.
     * ex: LoRA. 
   - Additive methods
     * keeping all of the original LLM weights frozen and introducing new trainable components.
     * Two main approaches
       + Adapter methods add new trainable layers to the architecture of the model, typically inside the encoder or decoder components after the attention or feed-forward layers.
       + Soft prompt methods, on the other hand, keep the model architecture fixed and frozen, and focus on manipulating the input to achieve better performance. This can be done by adding trainable parameters to the prompt embeddings or keeping the input fixed and retraining the embedding weights. ex: prompt tuning
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/e21aee417db84221b4b3921a9f6ec3ee53196a0d/week2_screenshots/0088.jpg)

## PEFT techniques 1: LoRA
1. A small recap of transformer architecture
   - The input prompt is turned into tokens, which are then converted to embedding vectors and passed into the encoder and/or decoder parts of the transformer.
   - In both of these components, there are two kinds of neural networks; self-attention and feedforward networks.
   - The weights of these networks are learned during pre-training.
   - After the embedding vectors are created, they're fed into the self-attention layers where a series of weights are applied to calculate the attention scores.
   - During full fine-tuning, every parameter in these layers is updated.
3. LoRA (Low-rank Adaptation)
   - is a parameter-efficient fine-tuning technique of the reparameterization category. 
   - Fine-tuning by freezing all of the original model parameters and then injecting a pair of rank decomposition matrices alongside the original weights.
   - The dimensions of the smaller matrices are set so that their product is a matrix with the same dimensions as the weights they're modifying.
   - You then keep the original weights of the LLM frozen and train the smaller matrices using the same supervised learning process.
   - For inference, the two low-rank matrices are multiplied together to create a matrix with the same dimensions as the frozen weights. You then add this to the original weights and replace them in the model with these updated values.
   - You now have a LoRA fine-tuned model that can carry out your specific task. Because this model has the same number of parameters as the original, there is little to no impact on inference latency.
   - Researchers have found that applying LoRA to just the self-attention layers of the model is often enough to fine-tune for a task and achieve performance gains. However, in principle, you can also use LoRA on other components like the feed-forward layers. But since most of the parameters of LLMs are in the attention layers, you get the biggest savings in trainable parameters by applying LoRA to these weights matrices.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/278aeb51beac08ca457d59e703b5448ddc9c65e6/week2_screenshots/0094.jpg)
4. You can often perform LoRA with a single GPU and avoid the need for a distribute cluster of GPUs.
5. Since the rank-decomposition matrices are small, you can fine-tune a different set for each task and then switch them out at inference time by updating the weights. The memory required to store these LoRA matrices is very small. So, you can use LoRA to train for many tasks.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/8bee1fca3a20fdcb9762307abc03d07b6acc5850/week2_screenshots/0098.jpg)
6. The ROUGE metrics for full vs. LoRA fine-tuning
   - Although FLAN-T5 is a capable model, it can still benefit from additional fine-tuning on specific tasks.
   - The ROUGE1 score of LoRA fine-tuned model is a little lower than full fine-tuning. However, using LoRA for fine-tuning trained a much smaller number of parameters than full fine-tuning using significantly less compute, so the small trade-off in performance may be worth it.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/001b41bfc3b2b3ab21b61aba8c0af4e64b2320e9/week2_screenshots/0101.jpg)
7. How to choose the rank of the LoRA matrices
   - the smaller the rank, the smaller the number of trainable parameters, and the bigger the savings on compute. However, there are some issues related to model performance to consider.
   - In the paper that first proposed LoRA, you can see the results of different rank. (The bold values indicate the best scores that were achieved for each metric)
   - The authors found a plateau in the loss value for ranks greater than 16. -> Using larger LoRA matrices didn't improve performance -> 4~32 can provide a good trade-off between reducing trainable parameters and preserving performance.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2039fe847d07555b7580c3377bcb4c4abaad5e0b/week2_screenshots/0102.jpg)

## PEFT techniques 2: Soft prompts
1. Additive methods within PEFT aim to improve model performance without changing the weights at all.
2. With 'Prompt engineering', you work on the language of your prompt to get the completion you want.
   - The goal is to help the model understand the nature of the task you're asking it to carry out and to generate a better completion
   - Limitations for prompt engineering
     * It can require a lot of manual effort to write and try different prompts.
     * Limited by the length of the context window.
3. With 'Prompt tuning', you add additional trainable tokens to your prompt and leave it up to the supervised learning process to determine their optimal values.
   - The set of trainable tokens is called a soft prompt, and it gets prepended to embedding vectors that represent your input text.
   - The soft prompt vectors have the same length as the embedding vectors of the language tokens. And including somewhere about 20 and 100 virtual tokens can be sufficient for good performance.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2039fe847d07555b7580c3377bcb4c4abaad5e0b/week2_screenshots/0106.jpg)
   - The tokens that represent natural language are hard in the sense that they each correspond to a fixed location in the embedding vector space.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2039fe847d07555b7580c3377bcb4c4abaad5e0b/week2_screenshots/0107.jpg)
   - However, the soft prompts are not fixed discrete words of natural language. Instead, you can think of them as virtual tokens that can take on any value within the continuous multidimensional embedding space. And through supervised learning, the model learns the values for these virtual tokens that maximize performance for a given task.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2039fe847d07555b7580c3377bcb4c4abaad5e0b/week2_screenshots/0108.jpg)
   - In 'full fine tuning', the training data set consists of input prompts and output completions or labels. The weights of the large language model are updated during supervised learning.
   - In contrast with prompt tuning, the weights of the large language model are frozen and the underlying model does not get updated. Instead, the embedding vectors of the soft prompt gets updated over time to optimize the model's completion of the prompt.
   - Prompt tuning is a very parameter efficient strategy because only a few parameters are being trained. In contrast with the millions to billions of parameters in full fine tuning.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2039fe847d07555b7580c3377bcb4c4abaad5e0b/week2_screenshots/0110.jpg)
   - You can train a different set of soft prompts for each task and then easily swap them out at inference time. You can train a set of soft prompts for one task and a different set for another. To use them for inference, you prepend your input prompt with the learned tokens to switch to another task, you simply change the soft prompt. Soft prompts are very small on disk, so this kind of fine tuning is extremely efficient and flexible. You'll notice the same LLM is used for all tasks, all you have to do is switch out the soft prompts at inference time.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2039fe847d07555b7580c3377bcb4c4abaad5e0b/week2_screenshots/0111.jpg)
4. how well does prompt tuning perform?
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2039fe847d07555b7580c3377bcb4c4abaad5e0b/week2_screenshots/0112.jpg)
   - The authors compared prompt tuning to several other methods for a range of model sizes.
   - As you can see, prompt tuning doesn't perform as well as full fine tuning for smaller LLMs. However, as the model size increases, so does the performance of prompt tuning. And once models have around 10 billion parameters, prompt tuning can be as effective as full fine tuning and offers a significant boost in performance over prompt engineering alone.
5. One potential issue to consider is the interpretability of learned virtual tokens.
   - Because the soft prompt tokens can take any value within the continuous embedding vector space, the trained tokens don't correspond to any known token, word, or phrase in the vocabulary of the LLM.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2039fe847d07555b7580c3377bcb4c4abaad5e0b/week2_screenshots/0113.jpg)
   - However, an analysis of the nearest neighbor tokens to the soft prompt location shows that they form tight semantic clusters.-> The words closest to the soft prompt tokens have similar meanings. The words identified usually have some meaning related to the task, suggesting that the prompts are learning word like representations.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2039fe847d07555b7580c3377bcb4c4abaad5e0b/week2_screenshots/0114.jpg)
6. PEFT methods summary
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2039fe847d07555b7580c3377bcb4c4abaad5e0b/week2_screenshots/0115.jpg)












