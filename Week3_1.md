# Reinforcement learning from human feedback

## Introduction
1. RLHF
2. How to use LLMs as a reasoning engine and let it cause our own of routines to create an agent

## Aligning models with human values
1. Some LLM behaving badly beacuse LLM are trained on vast amounts of texts data from the Internet where such language appears frequently.
   - Toxic language
   - Agressive responses
   - Providing dangerous information
2. HHH: helpfulness, honesty, harmlessness
   - are a set of principles that guide developers in the responsible use of AI
3. Additional fine-tuning with human feedback helps to better align models with human preferences and to increase the HHH of the completions.
   This further training can also help to decrease the toxicity, often models responses and reduce the generation of incorrect information. 

## Reinforcement learning from human feedback (RLHF)
1. Researchers at OpenAI published a paper that explored the use of fine-tuning with human feedback to train a model to write short summaries of text articles.
   - a model fine-tuned on human feedback produced better responses than a pretrained model, an instruct fine-tuned model, and even the reference human baseline.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0b4e44aba10c0b727f50c16c63c5b354b50045a8/Week3_screenshots/0008.jpg)
2. A popular technique to finetune large language models with human feedback is called reinforcement learning from human feedback (RLHF).
   - RLHF uses reinforcement learning to finetune the LLM with human feedback data, resulting in a model that is better aligned with human preferences.
   - You can use RLHF to make sure that your model produces outputs that maximize usefulness and relevance to the input prompt.
   - Perhaps most importantly, RLHF can help minimize the potential for harm. You can train your model to give caveats that acknowledge their limitations and to avoid toxic language and topics.
3. One potentially exciting application of RLHF is the personalizations of LLMs, where models learn the preferences of each individual user through a continuous feedback process. This could lead to exciting new technologies like individualized learning plans or personalized AI assistants.
4. Reinforcement learning
   - a type of machine learning in which an agent learns to make decisions related to a specific goal by taking actions in an environment, with the objective of maximizing some notion of a cumulative reward.
   - In this framework, the agent continually learns from its experiences by taking actions, observing the resulting changes in the environment, and receiving rewards or penalties, based on the outcomes of its actions.
   - By iterating through this process, the agent gradually refines its strategy or policy to make better decisions and increase its chances of success.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0b4e44aba10c0b727f50c16c63c5b354b50045a8/Week3_screenshots/0010.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0b4e44aba10c0b727f50c16c63c5b354b50045a8/Week3_screenshots/0011.jpg)
5. Ex: tic-tac-toe
   - agent: a model or policy acting as a Tic-Tac-Toe player.
   - objective: win the game.
   - environment: the three by three game board,
   - the state at any moment, is the current configuration of the board.
   - The action space comprises all the possible positions a player can choose based on the current board state.
   - The agent makes decisions by following a strategy known as the RL policy. Now, as the agent takes actions, it collects rewards based on the actions' effectiveness in progressing towards a win.
   - The goal of reinforcement learning is for the agent to learn the optimal policy for a given environment that maximizes their rewards.
   - This learning process is iterative and involves trial and error.
   - Initially, the agent takes a random action which leads to a new state. From this state, the agent proceeds to explore subsequent states through further actions.
   - The series of actions and corresponding states form a playout, often called a rollout.
   - As the agent accumulates experience, it gradually uncovers actions that yield the highest long-term rewards, ultimately leading to success in the game.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0b4e44aba10c0b727f50c16c63c5b354b50045a8/Week3_screenshots/0012.jpg)
6. how to fine-tuning large language models with RLHF
   - the agent's policy that guides the actions is the LLM
   - objective: generate text that is perceived as being aligned with the human preferences. This could mean that the text is, for example, helpful, accurate, and non-toxic.
   - The environment is the context window of the model, the space in which text can be entered via a prompt.
   - The state that the model considers before taking an action is the current context. That means any text currently contained in the context window.
   - The action here is the act of generating text. This could be a single word, a sentence, or a longer form text, depending on the task specified by the user.
   - The action space is the token vocabulary, meaning all the possible tokens that the model can choose from to generate the completion. 
7. How an LLM decides to generate the next token in a sequence, depends on the statistical representation of language that it learned during its training.
   - At any given moment, the action that the model will take, meaning which token it will choose next, depends on the prompt text in the context and the probability distribution over the vocabulary space.
   - The reward is assigned based on how closely the completions align with human preferences.
8. Given the variation in human responses to language, determining the reward is more complicated than in the Tic-Tac-Toe example.
   - One way you can do this is to have a human evaluate all of the completions of the model against some alignment metric, such as determining whether the generated text is toxic or non-toxic.
   - This feedback can be represented as a scalar value, either a zero or a one.
   - The LLM weights are then updated iteratively to maximize the reward obtained from the human classifier, enabling the model to generate non-toxic completions.
   - However, obtaining human feedback can be time consuming and expensive.
   - As a practical and scalable alternative, you can use an additional model, known as the reward model, to classify the outputs of the LLM and evaluate the degree of alignment with human preferences.
   - You'll start with a smaller number of human examples to train the secondary model by your traditional supervised learning methods.
   - Once trained, you'll use the reward model to assess the output of the LLM and assign a reward value, which in turn gets used to update the weights off the LLM and train a new human aligned version.
9. Exactly how the weights get updated as the model completions are assessed, depends on the algorithm used to optimize the policy.
10. Note that in the context of language modeling, the sequence of actions and states is called a rollout, instead of the term playout that's used in classic reinforcement learning.
11. The reward model is the central component of the reinforcement learning process. It encodes all of the preferences that have been learned from human feedback, and it plays a central role in how the model updates its weights over many iterations.

## RLHF: Obtaining feedback from humans
1. Steps:
   - The first step in fine-tuning an LLM with RLHF is to select a model to work with and use it to prepare a data set for human feedback.
   - The model you choose should have some capability to carry out the task you are interested in, whether this is text summarization, question answering or something else.
   - In general, you may find it easier to start with an instruct model that has already been fine tuned across many tasks and has some general capabilities.
   - You'll then use this LLM along with a prompt data set to generate a number of different responses for each prompt.
   - The prompt dataset is comprised of multiple prompts, each of which gets processed by the LLM to produce a set of completions.
   - The next step is to collect feedback from human labelers on the completions generated by the LLM. This is the human feedback portion of reinforcement learning with human feedback.
2. Collect human feedback
   - First, you must decide what criterion you want the humans to assess the completions on. This could be any of the issues discussed so far like helpfulness or toxicity.
   - Once you've decided, you will then ask the labelers to assess each completion in the data set based on that criterion.
   - The task for your labelers is to rank the three completions in order of helpfulness(or anything else) from the most helpful to least helpful.
   - This process then gets repeated for many prompt completion sets, building up a data set that can be used to train the reward model that will ultimately carry out this work instead of the humans.
   - The same prompt completion sets are usually assigned to multiple human labelers to establish consensus and minimize the impact of poor labelers in the group.
3. The clarity of your instructions can make a big difference on the quality of the human feedback you obtain. Labelers are often drawn from samples of the population that represent diverse and global thinking.
   - The following photo is an example set of instructions written for human labelers. This would be presented to the labeler to read before beginning the task and made available to refer back to as they work through the dataset.
   - The more detailed you make these instructions, the higher the likelihood that the labelers will understand the task they have to carry out and complete it exactly as you wish
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3ad2b1b55fd3903d84c73f1a849789ddd8841afe/Week3_screenshots/0018.jpg)
4. Once your human labelers have completed their assessments off the Prom completion sets, you have all the data you need to train the reward model. Which you will use instead of humans to classify model completions during the reinforcement learning finetuning process. Before you start to train the reward model, however, you need to convert the ranking data into a pairwise comparison of completions.
   - In other words, all possible 'pairs of completions' from the available choices to a prompt should be classified as 0 or 1 score.
   - For each pair, you will assign a reward of 1 for the preferred response and a reward of 0 for the less preferred response.
   - Then you'll reorder the prompts so that the preferred option comes first. This is an important step because the reward model expects the preferred completion, which is referred to as Yj first.
   - Note that while thumbs-up, thumbs-down feedback is often easier to gather than ranking feedback, ranked feedback gives you more prompt completion data (you will have N choose two combinations) to train your reward model. 

## RLHF: Reward model
1. The reward model will effectively take place off the human labeler and automatically choose the preferred completion during the RLHF process.
   - THe reward model is usually also a language model.
   - For a given prompt X, the reward model learns to favor the human-preferred completion y_ j, while minimizing the log sigmoid off the reward difference, r_j-r_k.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/b40ecfe7fc44353b7d3109bd0411f1b08cd9cc14/Week3_screenshots/0021.jpg)
2. Once the model has been trained on the human rank prompt-completion pairs, you can use the reward model as a binary classifier to provide a set of logics across the positive and negative classes.
   - Logics are the unnormalized model outputs before applying any activation function. Let's say you want to detoxify your LLM, and the reward model needs to identify if the completion contains hate speech.
   - Let's say you want to detoxify your LLM, and the reward model needs to identify if the completion contains hate speech. In this case, the two classes would be notate, the positive class that you ultimately want to optimize for and hate the negative class you want to avoid.
   - The largest value of the positive class is what you use as the reward value in LLHF. Just to remind you, if you apply a Softmax function to the logits, you will get the probabilities. The first example shows a good reward for non-toxic completion and the second example shows a bad reward being given for toxic completion. 
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/b40ecfe7fc44353b7d3109bd0411f1b08cd9cc14/Week3_screenshots/0023.jpg)

## RLHF: Fine-tuning with reinforcement learning
1. how to use the reward model in the reinforcement learning process to update the LLM weights?
   - start with a model that already has good performance on your task of interests -> work to align an instruction fine-tuned LLM.
   - First, pass a prompt from your prompt dataset to the instruct LLM, which then generates a completion.
   - Next, you sent this completion, and the original prompt to the reward model as the prompt completion pair.
   - The reward model evaluates the pair based on the human feedback it was trained on, and returns a reward value -> A higher value such as 0.24 as shown here represents a more aligned response. A less aligned response would receive a lower value, such as negative 0.53.
   - pass the reward value for the prompt completion pair to the reinforcement learning algorithm to update the weights of the LLM, and move it towards generating more aligned, higher reward responses. Let's call this intermediate version of the model 'the RL updated LLM'.
   - These series of steps together forms a single iteration of the RLHF process.
   - These iterations continue for a given number of epics, similar to other types of fine tuning. Here you can see that the completion generated by the RL updated LLM receives a higher reward score, indicating that the updates to weights have resulted in a more aligned completion.
   - If the process is working well, you'll see the reward improving after each iteration as the model produces text that is increasingly aligned with human preferences.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d40df2b4c39ff26184e22b7128bcd6c7732072a2/Week3_screenshots/0025.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d40df2b4c39ff26184e22b7128bcd6c7732072a2/Week3_screenshots/0026.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d40df2b4c39ff26184e22b7128bcd6c7732072a2/Week3_screenshots/0027.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d40df2b4c39ff26184e22b7128bcd6c7732072a2/Week3_screenshots/0028.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d40df2b4c39ff26184e22b7128bcd6c7732072a2/Week3_screenshots/0029.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d40df2b4c39ff26184e22b7128bcd6c7732072a2/Week3_screenshots/0030.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d40df2b4c39ff26184e22b7128bcd6c7732072a2/Week3_screenshots/0031.jpg)
2. You will continue the iterative process until your model is aligned based on some evaluation criteria.
   - for example, reaching a threshold value for the helpfulness you defined. You can also define a maximum number of steps, for example, 20,000 as the stopping criteria.
3. The 'RL algorithm' shown in the photo is the algorithm that takes the output of teh reward model and uses it to update the LLM model weights so that the reward score increases over time.
   - There are several different algorithms that you can use for this part of the RLHF process
   - A popular choice is 'proximal policy optimization' (PPO)

## Optional video: Proximal policy optimization
1. PPO (Proximal policy optimization)
   - a powerful algorithm for solving reinforcement learning problems
   - PPO optimizes a policy, in this case the LLM, to be more aligned with human preferences.
   - Over many iterations, PPO makes updates to LLM. The updates are small and within a bounded region, resulting in an updated LLM that is close to the previous version, hence the name Proximal Policy Optimization. Keeping the changes within this small region result in a more stable learning.
2. PPO's goal is to update the policy so that the reward is maximized. How this works in the specific context of LLM?
   - Start PPO with your initial instruct LLM, then at a high level, each cycle of PPO goes over two phases.
3. Phase I
   - In Phase I, the LLM, is used to carry out a number of experiments, completing the given prompts. These experiments allow you to update the LLM against the reward model in Phase II, remember that the reward model captures the human preferences. For example, the reward can define how helpful, harmless, and honest the responses are. The expected reward of a completion is an important quantity used in the PPO objective.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/81df39a11da3ef7bc2f21104d0a0be10be635132/Week3_screenshots/0035.jpg)
   - We estimate this quantity through a separate head of the LLM called the value function. Let's have a closer look at the value function and the value loss. Assume a number of prompts are given. First, you generate the LLM responses to the prompts, then you calculate the reward for the prompt completions using the reward model.
   - For example, the first prompt completion shown in the photo might receive a reward of 1.87. The next one might receive a reward of -1.24, and so on. You have a set of prompt completions and their corresponding rewards.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/81df39a11da3ef7bc2f21104d0a0be10be635132/Week3_screenshots/0036.jpg)
   - The value function estimates the expected total reward for a given State S. In other words, as the LLM generates each token of a completion, you want to estimate the total future reward based on the current sequence of tokens.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/81df39a11da3ef7bc2f21104d0a0be10be635132/Week3_screenshots/0037.jpg)
   - You can think of this as a baseline to evaluate the quality of completions against your alignment criteria. Let's say that at this step of completion, the estimated future total reward is 0.34. With the next generated token, the estimated future total reward increases to 1.23. The goal is to minimize the value loss that is the difference between the actual future total reward in this example, 1.87, and its approximation to the value function, in this example, 1.23. The value loss makes estimates for future rewards more accurate.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/81df39a11da3ef7bc2f21104d0a0be10be635132/Week3_screenshots/0039.jpg)
   - The value function is then used in Advantage Estimation in Phase 2. 
4. Phase II
   - In Phase 2, you make a small updates to the model and evaluate the impact of those updates on your alignment goal for the model. The model weights updates are guided by the prompt completion, losses, and rewards.
   - PPO also ensures to keep the model updates within a certain small region called the trust region. This is where the proximal aspect of PPO comes into play. Ideally, this series of small updates will move the model towards higher rewards. The PPO policy objective is the main ingredient of this method. Remember, the objective is to find a policy whose expected reward is high. In other words, you're trying to make updates to the LLM weights that result in completions more aligned with human preferences and so receive a higher reward. The policy loss is the main objective that the PPO algorithm tries to optimize during training. 
5. PPO Phase 2: Calculate policy loss
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/81df39a11da3ef7bc2f21104d0a0be10be635132/Week3_screenshots/0041.jpg)



