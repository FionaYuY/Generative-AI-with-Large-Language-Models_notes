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



