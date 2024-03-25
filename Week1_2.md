# Week1-2 (LLM pre-training and scaling laws)
## Pre-training LLM
1. Considerations for choosing a model
   - Foundation model(Pre-trained LLM) vs. Train your own model(Custom LLM)
2. HuggingFace and PyTorch and some other developers have curated hubs where you can browse some models.
3. A really useful feature of these hubs is the inclusion of model cards, that describe important details including the best use cases, how it was trained, limitations...
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/7d0749ffc47f1086a0febec6f0a67915ff9c8a7d/screenshots%20of%20lecture%20slides/0096.jpg)
4. Variance of the transformer model architecture are suited to different language tasks, largely because of differences in how the models are trained.
5. 'Pre-training': the initial training process for LLMs.
   - When the model learns from vast amounts of unstructured textual data
   - In this self-supervised learning step, the model internalizes the patterns and structures present in the language.
   - These patterns then enable the model to complete its training objective.
   - During pre-training, the model weights get updated to minimize the loss of the training objective.
   - The encoder generates an embedding or vector representation for each token.
   - Requires a large amount of compute and the use of GPUs
  ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/ca8d75ca5c1f0afe846b1ef5a5e88bea7f807829/screenshots%20of%20lecture%20slides/0098.jpg)
6. Encoder-only models
     - also known as Autoencoding models
     - they are pre-trained using masked language modeling.
     - Tokens in the input sequence or randomly mask, and the training objective is to predict the mask tokens in order to reconstruct the original sentece. -> also called a 'denoising objective'.
     - is built bi-directional representations of the input sequence, meaning that model has an understanding of the full context of a token and not just of the words come before.
  ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d58c2b6f29599d393c1dfd05e9fa3828ebe17c16/screenshots%20of%20lecture%20slides/0100.jpg)
  ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d58c2b6f29599d393c1dfd05e9fa3828ebe17c16/screenshots%20of%20lecture%20slides/0101.jpg)
7. Decoder-only models:
   - also known as Autoregressive models
   - they are pre-trained using causal language modeling.
   - The training objective is to predict the next token based on the previous sequence of tokens. Predicting the next token is sometimes called 'full language modeling'
   - Decoder-based autoregressive models, mask the input sequence and can only see the input tokens leading up to the token in question.
   - The model has no knowledge of the end of the sentence.
   - Unidirectional
   - By learning to predict the next token from a vast number of exampkes, the model builds up a statistical representation of language.
   - Decoder-only models are often used for text generation, although larger decoder-only models show strong zero-shot inference abilities, and can often perform a range of tasks well
  ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/385bd1a05634bb93bae1943258a613ad756a8833/screenshots%20of%20lecture%20slides/0102.jpg)
  ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/385bd1a05634bb93bae1943258a613ad756a8833/screenshots%20of%20lecture%20slides/0103.jpg)
  8. Encoder-decoder models:
     - also known as Sequence-to-Sequence model
     - The exact detail of the pre-training objective vary from model to model
     - T5
       * pre-trains the encoder using span corruption, which masks random sequences of input tokens. Those mass sequences are then replaced with a unique Sentinel token, show here as x.
       * Sentinel tokens are special tokens added to the vocabulary, but do not correspond to any actual word form the input text.
       * The decoder is then tasked with reconstructing the mask token sequence auto-regressively.
       * The output is the Sentinel token followed by the predicted tokens.
    ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/61ef6015d1af0a976e03e252a5041cecf5cff889/screenshots%20of%20lecture%20slides/0104.jpg)
    ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/61ef6015d1af0a976e03e252a5041cecf5cff889/screenshots%20of%20lecture%20slides/0104.jpg)
     - You can use sequence-to-sequence models for translation, summarization, and question-answering. They are generally useful in cases where you have a body of texts as both input and output.
  9. Comparison of the different model architectures and the targets off the pre-training objectives.
     ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/ffb9eb8e58b2dabd3f370408116a486cb0a08cda/screenshots%20of%20lecture%20slides/0106.jpg)
  10. The larger a model, the more likely it is to work as you needed to without additional in-context learning or further training.
      ![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/ffb9eb8e58b2dabd3f370408116a486cb0a08cda/screenshots%20of%20lecture%20slides/0108.jpg)
       
## Computational challenges of training LLMs
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0110.jpg)
1. 'Out of memory': Because most LLMs are huge, and require a ton of memeory to store and train all their parameters.
2. 'CUDA': Compute Unified Device Architecture, is a collection of libraries and tools developed for Nvidia GPUs.
3. Libraries such as PyTorch and TensorFlow use CUDA to boost performance on metrics multiplication and other operations common to deep learning
4. Approximate GPU RAM needed to store 1B parameters
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0111.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/81a4208e619e56ecb26fc2041073b35b0ec1054a/screenshots%20of%20lecture%20slides/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%20(666).png)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0113.jpg)
6. 'Qunatization'
   - one technique that you can use to reduce the memory.
   - you reduce the memory required to store the weights of your model by reducting their precision from 32-bit floating point numbers to 16-bit floating point numbers , or 8-bit integer numbers.
   - The corresponding data types are FP32 for 32-bit full position, FP16, or Bfloat16 for 160bit half precision, and int8
   - By default, model weights, activations, and other model parameters are stored in FP32
   - Quantization statistically projects the original FP32 into a lower precision space, using scaling factors calculated based on the range of the original FP32. 
![image]((https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0114.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0115.jpg)
![image]((https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0116.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0117.jpg)
   - Many LLMs, including flat-T5, have been pre-trained with BFloat16
   - BFloat16 is a hybrid between FP16 and FP32. BFloat16 is often described as a truncated 32-bit float, as it captures the full dynamic range of FP32, that uses only 16-bits. Not only save memory, but also increases model performance by speeding up calculations. However, BF16 is not well suited for integer calculations, but these are relatively rare in deep learning.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0118.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0119.jpg)
9. Note in all this cases you still have a model with one billion parameters. As you can see, the circles representing the models are the same size. Quantization will give you the same degree of savings when it comes to training.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0120.jpg)
10. As model scale beyond a few billion parameters, you need to turn to distributed computeing techniques while you train your model across multiple GPUs.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0122.jpg)









    
