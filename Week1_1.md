# Week 1-1 (Introduction to LLMs and the generative AI project lifecycle)

The official lecture notes is on : https://community.deeplearning.ai/t/genai-with-llms-lecture-notes/361913

## Course Introduction 
1. Week 1
   - Transformer architecture
   - in-context learning
   - tuning parameters
2. Week 2
   - pre-trained model
   - instruction fine tuning
3. Week 3
   - align the output of language models with human values
   - Increase helpfullness, decrease potential harm and toxicity

## Introduction - Week 1
1. Transformer architecture, self-attention, multi-headed self-attention mechanism
2. Generative AI project lifecycle

## Generative AI & LLMs
1. Generative AI is a subset of traditional machine learning.
2. The following photos are a collection of foundation models, sometimes called 'base models'.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/51907c27bf39acd139b3aa5cd88b6fcfff0420e9/screenshots%20of%20lecture%20slides/0008.jpg)
3. The more parameters a model has, the more memory, and the more sophisticated the tasks it can perform
4. Generative AI models are being created for 'multiple modalities'. ("Multiple modalities" in the context of generative AI models refer to the ability of these models to generate content across different sensory modalities or types of data. )
For instance, a generative AI model might be able to generate not just text, but also images, audio, or even video. This capability allows the model to create diverse content across various formats, making it versatile in its creative output.
5. LLMs are able to take natuaral language or human written instructions and perform taskes much as a human would.
6. The text you pass to an LLM is 'prompt', and the space or memory that is available to the prompt is 'context window', and the output of the model is 'completion'. The act of using the model to generate text is 'inference'.
7. The 'completion' is comprised of the text contained in the original prompt, followed by the generated text.

## LLM use cases and tasks
1. 'Next word prediction' is the concept behind a number of different capabilities, such as a basic chatbot.
2.  Use cases
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3391b9a307d6699c8ce0395b89ca29362fcb720/screenshots%20of%20lecture%20slides/0016.jpg)
3. 'Name entity recognition': you can ask the model to identify all of the people and places identified in a news article.
4. You can also augmenting LLMs by connecting them to external data sources, or using them to invoke external APIs. By doing this, you can provide the model with information it doesn't know from its pre-training.
5. Smaller models can be finetuned to perform well on specific focused tasks.

## Generating text with RNNs
1. Previous generations of language models made use of 'Recurrent neural networks' (RNN).
2. RNN were limited by the amount of compute and memory needed to perform well on generated tasks.
3. With just one previous words seen by the model, the prediction can't be very good.
4. Even though you scale the model, it still hasn't seen enough of the input.
5. Understanding language can be challenging.
6. 'Homonyms'(同音異義): ex, I took my money to the 'bank'.(銀行or river bank?)
7. 'Syntactic ambiguity': ex, The teacher taught the student with the book. -> teacher's book? student's book?
8. 'Attention is all you need': transformer architecture -> everything changed.
9. 'Transformer architecture'
    - Scale efficiently
    - Parallel process
    - Attention to input meaning
  
## Transformers architecture
1. The power of the transformer architecture lies in its ability to learn the relevance and context of all of the words in a setence.
   Not just to each word next to its neighbor, but to every other word in the setence.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3391b9a307d6699c8ce0395b89ca29362fcb720/screenshots%20of%20lecture%20slides/0032.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3391b9a307d6699c8ce0395b89ca29362fcb720/screenshots%20of%20lecture%20slides/0033.jpg)
2. To apply attention weights to those relationships so that the model learns the relevance of each word to each other word.
3. 'Attention map'
4. 'Self-attention' 可參考 https://github.com/FionaYuY/MachineLearning2021_HungYiLee/blob/b3198aa71409a9c217e1f5c1b1b5976f4e081f0e/Self-attention_1.md
5. The ability to learn a tension in this way (Self-attention) across the whole input significantly approves the model's ability to encode language.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3391b9a307d6699c8ce0395b89ca29362fcb720/screenshots%20of%20lecture%20slides/0035.jpg)
6. The transformer architecture is split into 2 parts: encoder and decoder.
7. Before passing texts into the modle, you have to 'tokenize' the word.
8. 'Tokenize': to convert the words into numbers with each number representing a position in a dictionary of all the possible words that the model can work with.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3391b9a307d6699c8ce0395b89ca29362fcb720/screenshots%20of%20lecture%20slides/0040.jpg)
10. Once you select a tokenizer to train the model, you have to use the same tokenizer when you generate text.
11. You can pass the tokenized input into the 'embedding layer'.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3391b9a307d6699c8ce0395b89ca29362fcb720/screenshots%20of%20lecture%20slides/0042.jpg)
12. 'Embedding layer': is a trainable vector embedding space, a high dimensional space where each token is represented as a vector and occupies a unique location within that space.
13. Each token id in the vocabulary is matched to a multi-dimensional vector, and the intuition is that these vectors learn to encode the meaning and context of individual tokens in the input sequence.
14. Embedding vector spaces have been used in natural language processing for some time, such as Word2vec.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/981c797555f5ae70b105c06c3124ab1e4e5246a4/screenshots%20of%20lecture%20slides/0043.jpg)
15. Each word has been matched to a token id -> each token is mapped into a vector
16. The following is a demonstrate with 3 dimensional space. You can see now how you can relate words that are located close to each other in the embedding space, and how yo can calculate the distance betweens the word as an angle, which gives the model the ability to mathematically understand language.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/c8e1ab5c3c96b68a6106bba080e70bffdad9dadb/screenshots%20of%20lecture%20slides/0046.jpg)
17. Notice that in the original transformer paper, the vector size is actually 512.'
18. As you add the token vectors into the base of the encoder or the decoder, you also add positonal encoding. The model processes each of the input tokens in parallel, so by adding positional encoding, you preserve the information about the word order and don't lose the relevance of the positon of the word in the sentence.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/547fac9c8d437a82f91b144733092578451ea624/screenshots%20of%20lecture%20slides/0047.jpg)
19. Once you've summed the input tokens and the positional encodings, you pass the resulting vectors to the self-attention layer. The model will analyzes the relationships bettween the tokens in your input sequence.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/9d53bbd4dcd21f90de5018fe0bfac8eb5e56d8bf/screenshots%20of%20lecture%20slides/0048.jpg)
21. The self-attention weights that are learned during training and stored in these layers reflect the importance of each word in that input sequence to all other words in the sequence. This doesn't happen just once, the transformer architecute actually has multi-headed self-attention. This means that multiple sets of self-attnetion weights or heads are learned in parallel independently to each other.
22. The number of attention heads included in the attention layer varies from model to model, range in 12-100 are common.
23. Each self-attention head will learn a different aspect of language.
24. The weights of each head are randomly initialized and given sufficient training data and time, each will learn different aspects of language.
25. After all of the attention weights have been applied to your input data, the output is processed through a fully-connected feed-forward network. The output of this layer is a vector of logits proportional to the probability score for each and every token in the tokenizer dictionary. You can pass these logits to a final softmax layer, where there are normalized into a probability score for each word. 
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/9d53bbd4dcd21f90de5018fe0bfac8eb5e56d8bf/screenshots%20of%20lecture%20slides/0050.jpg)

## Generating text with transformers
1. Translation: sequence-to-sequence task.
   - Task: translate the French phrase into English.
     ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0f8c104863f5af4b69927b3752301d3e8097e9b1/screenshots%20of%20lecture%20slides/0052.jpg)
   - First, tokenize the input using the same tokenizer that was used to train the network
     ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0f8c104863f5af4b69927b3752301d3e8097e9b1/screenshots%20of%20lecture%20slides/0053.jpg)
   - These tokens are then added into the input on the encoder side of the network, passed through embedding layer, and then fed into the multi-headed attention layers.
      ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0f8c104863f5af4b69927b3752301d3e8097e9b1/screenshots%20of%20lecture%20slides/0054.jpg)
   - The outputs of the multi-headed attention layers are fed through a feed-forward network to the output of the encoder.
   - Now, the data that leaves the encoder is a deep representation of the structure and meaning of the input sequence.
   - This representation is inserted into the middle of the decoder to influence the decoder's self-attention mechanisms.
     ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0f8c104863f5af4b69927b3752301d3e8097e9b1/screenshots%20of%20lecture%20slides/0055.jpg)
   - A start of sequence token is added to the input of the decoder.
   - This triggers the decoder to predict the next token, which it does based on the contextual understanding that is being provided from the encoder.
     ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0f8c104863f5af4b69927b3752301d3e8097e9b1/screenshots%20of%20lecture%20slides/0056.jpg)
   - The output of the decoder's self-attention layers gets passed through the decoder feed-forward network and through a final softmax output layer. At this point, we have our first token.
     ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0f8c104863f5af4b69927b3752301d3e8097e9b1/screenshots%20of%20lecture%20slides/0057.jpg)
   - Continue the loop, passing the output token back to the input to trigger the generation of the next token, until the model predicts an end-of-sequence token.
     ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0f8c104863f5af4b69927b3752301d3e8097e9b1/screenshots%20of%20lecture%20slides/0058.jpg)
   - At this point, the final sequence of tokens can be detokenized into words, and that's your output.
     ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/0f8c104863f5af4b69927b3752301d3e8097e9b1/screenshots%20of%20lecture%20slides/0059.jpg)
2. There are many ways to use the output from the softmax layer to predict the next token, these can influence how creative the generated text is.
3. In conclusion, the transformer architecture consists of encoder and decoder component.
   ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/e997f2ff98b6edc5ffe281ea20ccd831ad8ebe31/screenshots%20of%20lecture%20slides/0061.jpg)
4. "Sequence-to-sequence" (often abbreviated as seq2seq)
   - refers to a type of model architecture used in machine learning, particularly in tasks involving sequences of data.
   - In a sequence-to-sequence model, the input and output are both sequences of data, which can vary in length. This architecture is commonly used in tasks such as machine translation, text summarization, and speech recognition, among others.
   - The basic idea behind sequence-to-sequence models is to have an encoder-decoder architecture. The encoder takes the input sequence and processes it into a fixed-dimensional vector, often called the "context" or "thought vector," which contains a condensed representation of the input sequence. The decoder then takes this context vector and generates the output sequence step by step.
   - Sequence-to-sequence models are often implemented using recurrent neural networks (RNNs) or more advanced variants like long short-term memory networks (LSTMs) or gated recurrent units (GRUs). These models have shown success in various natural language processing tasks where the input and output are sequences of variable lengths.
5. Encoder Only models
   - also works as sequence-to-sequence models, but without further modification, the input sequence and the output sequence are the same length. 
   - less common these days, but by adding additional layers to the architecture, you can train encoder-only models to perform classification tasks such as sentiment analysis.
   - ex: BERT
6. Encoder-decoder models
   - perform well on sequence-to-sequence tasks such as translation, where the input sequence and the output sequence can be in different lengths.
   - By scaling and training this kind of model, it can perform general text generation tasks.
   - ex: BART 
7. Decoder-only models
   - some of the most commonly used today
   - once they're scaled, their capabilities have grown.
   - These models can now generalize to most tasks.
   - ex: GPT family, BLOOM, Jurrassic, LLaMA...
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/bd52098e56ffab4397ab4e7b66abfdefeb0f2ea4/screenshots%20of%20lecture%20slides/0062.jpg)

## Tranformers: Attention is all you need
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/92270cab1830b65ccd4d7033b04faae1fccff584/screenshots%20of%20lecture%20slides/transformer_image.png)
1. The Transformer model uses self-attention to compute representations of input sequences, which allows it to capture long-term dependencies and parallelize computation effectively.
2. The Transformer architecture consists of an encoder and a decoder, each of which is composed of several layers. Each layer consists of two sub-layers: a multi-head self-attention mechanism and a feed-forward neural network.
3. The multi-head self-attention mechanism allows the model to attend to different parts of the input sequence, while the feed-forward network applies a point-wise fully connected layer to each position separately and identically. 
4. The Transformer model also uses residual connections and layer normalization to facilitate training and prevent overfitting.
5. The authors introduce a positional encoding scheme that encodes the position of each token in the input sequence, enabling the model to capture the order of the sequence without the need for recurrent or convolutional operations.

## Prompting and promt engineering
1. 'Prompt engineering': you may have to revise the language in your prompt or the way that it's written several times to get the model to behave in the way that you waant.
2. 'In-context learning': Providing examples inside the context window.
3. In-context leraning (ICL) - zero shot inference (not example provided)
   - involves providing an example question with answer followed by a second question to be answered by the LLM.
   - Largest of the LLM are good at this
   - Smaller model aren't good at this
5. In-context learning (ICL) - one shot inference (provide one example)
6. In-context learning (ICL) - few shot inference (provide multiple examples)
7. Generally, your model isn't performing well even if you provide 5~6 examples, you should fine-tune the model.
8. There's a limit in context window

## Generative configuration
1. Each model exposes a set of configuration parameters that can influence the model's output during inference.
2. Different from the training parameters which are learned during training time. These configuration parameters are invoked at inference time.
3. 'Max new tokens': The model of tokens that the model will generate
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/e5b4b44613424e4ddf96c500b4756e295487ca62/screenshots%20of%20lecture%20slides/0075.jpg)
4. The output from the transformer's softmax layer is a probability distribution across the entire dictionary of words that the model uses.
5. Most LLM by default will operate with 'greedy decoding'.
6. 'Greedy decoding': the simplest form of next-word prediction, where the model will always choose the word with the highest probability
   - This method can work very well for short generation but is susceptible to repeated words or repeated sequences of words.
7. 'Random sampling': the easiest way to introduce some variability.
   - The model chooses an output word at random using the probability distribution to weight the selection.
   - By using random sampling, it reduces the likelihood that words will be repeated.
   - Depending on the setting, there is a possibility that the output may be too creative, producting words that cause the generation to wander off into topics or words that just don't make sense.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/b21f783b52c5181b4d301428f65107a58f4bd438/screenshots%20of%20lecture%20slides/0076.jpg)
8. Top p and top k are sampling techniques that we can use to help limit the random sampling and increase the chance that the output will be sensible.
9. 'Top k': to limit the oprions while still allowing some variability, you can specify a top k value which instructs the model to choose from only the k tokens with the highest probability.
   - For example, k=3, which restricts the model to choose from the highest three options. The model then selects from these options using the probability weighting.
   - This moethod can help the model have some randomness while preventing the selection of highly improbable completion words
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/817d030cbf6019a64117d423b45c84526eb2a03a/screenshots%20of%20lecture%20slides/0078.jpg)
10. 'Top p': limit the random sampling to the predictions whose combined probabilities (cumulative probabilities) do not exceed p.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/f0ddf0dfc8876ba18cc79810e509ae71fc758452/screenshots%20of%20lecture%20slides/0079.jpg)
12. Top k, specify the number of the tokens to randomly choose from. Top p, specify the total probability that you want the model to choose from.
13. 'Temperature': influences the shape of the probability distribution that the model calculates for the next token.
    - The higher the temperature, the higher the randomness.
    - The temperature value is a scaling factor that's applied within the final softmax layer of the model that impacts the shape of the probability diestribution of the next token.
14. In contrast to the top p and top k, changing the temperaure actually alters the predictions that the model will make.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/fd8ed95ae8d637cbeb65b52fd3a71fae2b0fd610/screenshots%20of%20lecture%20slides/0081.jpg)
15. if temperture=1, this will leave the softmax function as default, and the unaltered probability distribution will be used.

## Genrative AI project lifecycle
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/c106d0c11df1eea08bde0f0467e982d0b2d6b66e/screenshots%20of%20lecture%20slides/0083.jpg)
1. Scope
   - Define the scope as accurately and narrowly as you can.
   - carry out many different tasks vs. specific tasks only
2. Select
   - Train your own model from scratch vs. work with an existing base model
3. Adapt and align model
   - prompt engineering can sometimes be enough to get your model to perform well, so you can start by trying in-context learning.
   - Once your model doesn't perform as well as you need -> fine-tuning
   - As models become more capable, it's becoming increasingly important to ensure that they behave well and in a way that is aligned with human preferences in deployment.
   - Evaluation
   - Note that this adapt and align stage can be highly iterative.
4. Application integration
   - optimize your model for deployment
   - consider any additional infrastructure that your application will require to work well.
  









   



