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
     + Causal Language Modeling is a way to teach a language model to predict the next word in a sentence based on the words that come before it. It's like completing a sentence where you always look back at the previous words to decide what comes next, but you never peek ahead.
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
  11. "Span corruption" is a technique used in some pre-training methods for language models, specifically in the context of self-supervised learning. Unlike Masked Language Modeling (MLM), where individual tokens or words are masked or hidden from the model, span corruption involves masking or removing longer sequences of consecutive tokens or entire spans of text. Here's a bit more detail on the process:

   - A span of text, which could be a few consecutive words, is selected randomly from a sentence.
   - This span is then replaced with a special token (or tokens) indicating that a span has been removed.
   - The language model's task during training is then to predict the removed span, not just single words. This requires understanding and generating multiple words in the correct order, which can be more challenging than predicting individual tokens.
   - By predicting spans rather than individual words, the model may learn to understand and generate more coherent pieces of text, improving its ability to handle complex language tasks.

Span corruption forces the model to rely on a broader context, enhancing its ability to understand and generate longer sequences of text, which is beneficial for various natural language processing tasks.
## Computational challenges of training LLMs
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0110.jpg)
1. 'Out of memory': Because most LLMs are huge, and require a ton of memeory to store and train all their parameters.
2. 'CUDA': Compute Unified Device Architecture, is a collection of libraries and tools developed for Nvidia GPUs.
3. Libraries such as PyTorch and TensorFlow use CUDA to boost performance on metrics multiplication and other operations common to deep learning
4. Approximate GPU RAM needed to store 1B parameters
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a3ecf15266bcb68af6f4984263598c576a27e34e/screenshots%20of%20lecture%20slides/0111.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/81a4208e619e56ecb26fc2041073b35b0ec1054a/screenshots%20of%20lecture%20slides/%E8%9E%A2%E5%B9%95%E6%93%B7%E5%8F%96%E7%95%AB%E9%9D%A2%20(666).png)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/da9f2965a853248ab8f453aad528749b449b00d5/screenshots%20of%20lecture%20slides/0113.jpg)
6. 'Qunatization'
   - one technique that you can use to reduce the memory.
   - you reduce the memory required to store the weights of your model by reducting their precision from 32-bit floating point numbers to 16-bit floating point numbers , or 8-bit integer numbers.
   - The corresponding data types are FP32 for 32-bit full position, FP16, or Bfloat16 for 160bit half precision, and int8
   - By default, model weights, activations, and other model parameters are stored in FP32
   - Quantization statistically projects the original FP32 into a lower precision space, using scaling factors calculated based on the range of the original FP32. 
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/da9f2965a853248ab8f453aad528749b449b00d5/screenshots%20of%20lecture%20slides/0114.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/da9f2965a853248ab8f453aad528749b449b00d5/screenshots%20of%20lecture%20slides/0115.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/da9f2965a853248ab8f453aad528749b449b00d5/screenshots%20of%20lecture%20slides/0116.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/da9f2965a853248ab8f453aad528749b449b00d5/screenshots%20of%20lecture%20slides/0117.jpg)
   - Many LLMs, including flat-T5, have been pre-trained with BFloat16
   - BFloat16 is a hybrid between FP16 and FP32. BFloat16 is often described as a truncated 32-bit float, as it captures the full dynamic range of FP32, that uses only 16-bits. Not only save memory, but also increases model performance by speeding up calculations. However, BF16 is not well suited for integer calculations, but these are relatively rare in deep learning.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/da9f2965a853248ab8f453aad528749b449b00d5/screenshots%20of%20lecture%20slides/0118.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/da9f2965a853248ab8f453aad528749b449b00d5/screenshots%20of%20lecture%20slides/0119.jpg)
9. Note in all this cases you still have a model with one billion parameters. As you can see, the circles representing the models are the same size. Quantization will give you the same degree of savings when it comes to training.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/da9f2965a853248ab8f453aad528749b449b00d5/screenshots%20of%20lecture%20slides/0120.jpg)
10. As model scale beyond a few billion parameters, you need to turn to distributed computeing techniques while you train your model across multiple GPUs.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/da9f2965a853248ab8f453aad528749b449b00d5/screenshots%20of%20lecture%20slides/0122.jpg)

## Effiecient multi-GPU compute strategies
1. The first step in scaling model training is to distribute large data-sets across multiple GPUs and process these batches of data in parallel.
2. A popular implementation of this model replication technique is PyTorch's 'Distributed Data Parallel (DDP)'.
3. 'Distributed Data Parallel (DDP)'
   - DDP copyists your model onto each GPU and sends batches of data to each of the GPUs in parallel.
   - Each dataset is processed in parallel and then a synchronization step combines the results of each GPU, which in turn updates the model on each GPU, which is always identical across chips.
   - This implementation allows parallel computations across all GPUs that results in faster training.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/00c89e241da972e4f7e12ab1f66a1e8e91b3eaf6/screenshots%20of%20lecture%20slides/0125.jpg)
4. Note that DDP requires that your model weights and all of the additional parameters, gradients, and optimizer states that are needed for training, fit onto a single GPU.
   if your model is too big for this, you should try another technique called 'model sharding'. 
5. A popular implementation of model sharding is PyTorch's 'Fully Sharded Data Parallel (FSDP)'
6. Fully Sharded Data parallel (FSDP)
   - It is motivated by a paper 'ZeRO: Memory Optimizations Toward Training Trillion Parameter Models' from Microsoft.
   - 'ZeRO' stands for zero redundancy optimizer and the goal of ZeRO is to optimize memory by distributing or sharding model states across GPUs with ZeRO data overlap.
     This allows you to scale model training across GPUs when your model doesn't fit in the memory of a single chip.
   - Recap that the largest memory requirement was for the optimizer states, which take up twice as much space as the weights, followed by weights themselves and the gradients.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/66e50f126951109d2b10cee852f89e4f26518159/screenshots%20of%20lecture%20slides/0127.jpg)
   - One limitation of the model replication strategy is that you need to keep a full model copy on each GPU, which leads to redundatn memory consumption. You are storing the same numbers on every GPU.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/1ed8c6fe9e85dc894a31184ca00d1ff522ccd9e2/screenshots%20of%20lecture%20slides/0128.jpg)
   - ZeRO eliminates this redundancy by distributing also referred to as 'sharding' the model parameters, gradients, and optimizer states across GPUs instead of replicating them. At the same time, the communication overhead for a sinking model states stays close to that of the previously discussed DDP.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/1af52ade7109571bdc82f6b2cb1f247cfaa4b4ab/screenshots%20of%20lecture%20slides/0129.jpg)
7. ZeRO offers 3 optimization stages
   - ZeRO stage 1: shots only optimizer states across GPUs, which can reduce your memory footprint by up to a factor of four.
   - ZeRO stage 2: shots the gradients across chips. When applied together with stage 1, this can reduce your memory footprint by up to 8 times.
   - ZeRO stage 3: shots all components including the model parameters across GPUs. When applied together with stage 1 and 2, memory reduction is linear with a number of GPUs. ex: sharding across 64 GPUs could reduce your memory by a factor of 64
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/a6ac59dca4925dd48c2770fbe108e15e30866d30/screenshots%20of%20lecture%20slides/0130.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/2cca02753a9501611d5f06ba3443501ee19d1164/screenshots%20of%20lecture%20slides/0134.jpg)
8. In contratst to DDP, where each GPU has all of the model states required for processing each batch of data available locally, FSDP requires you to collect this data from all of the GPUs before the forward and backward pass. Each GPU requests data from the other GPUs on-demand to materialize the sharded data into unsharded data for the duration of the operation. After the operation, you release the unsharded non-local data back to the other GPUs as original sharded data. You can hoose to keep it for future operations during backward pass too. Note, this requires more GPU RAM again, this is a typical performance versus memory trade-off decision.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob./1bdc8e2137b40534c0036e27b3219748e834ff84/screenshots%20of%20lecture%20slides/0135.jpg)
10. In the final step after the backward pass, FSDP synchronizes the gradients across the GPUs in the same way they were for DDP.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/819a3eb3883fef5688d6fd85b31348db237a96f9/screenshots%20of%20lecture%20slides/0136.jpg)
11. Optionally, you can specify that FSDP offloads part of the training computation to CPUs to further reduce your GPU memory utilization.
12. To manage the trade-off between performance and memory utilization, you can configure the level of sharding using FSDP's sharding factor
    - sharding factor=1, removes the sharding and replicates the full model similar to DDP
    - sharding factor=maximum number of available GPU, you turn on full sharding. This has the most memory savings, but increases the communication volume between GPUs.
    - Any sharding factor in beween enables hybrid sharding 
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/b193211b29e0173cb71593c49c9355d80e8cfac1/screenshots%20of%20lecture%20slides/0137.jpg)
13. 1 teraflop corresponds to one trillion floating-point operations per second.
14. These tests were performed using a maximum of 512 NVIDIA A100 GPUs, each with 80 gigabytes of memory.
    - The first figure shows FSDP performance for different size T5 models.
    -  You can see the different performance numbers for FSDP, full sharding in blue, hybrid sharding in orange and full replication in green.
    -  For reference, DDP performance is shown in red.
    -  For the first 25 models with 611 million parameters and 2.28 billion parameters, the performance of FSDP and DDP is similar.
    -  If you choose a model size beyond 2.28 billion, such as T5 with 11.3 billion parameters, DDP runs into the out-of-memory error. FSDP on the other hand can easily handle models this size and achieve much higher teraflops when lowering the model's precision to 16-bit.
    -  The second figure shows 7% decrease in per GPU teraflops when increasing the number of GPUs from 8~512 for the 11 billion T5 model, plotted here using a batch size of 16 in orange and a batch size of eight in blue.
    -  As the model grows in size and is distributed across more and more GPUs, the increase in communication volume between chips starts to impact the performance, slowing down the computation.
    -  In summary, this shows that you can use FSDP for both small and large models and seamlessly scale your model training across multiple GPUs.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/b1a09f360cee4113461a1bf7aeac8cf9833c2621/screenshots%20of%20lecture%20slides/0138.jpg)

## Scaling laws and compute-optimal models
1. Scaling choices for pre-training
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/d46ec4a080a80380c0000602ad81eeaa72cf1116/screenshots%20of%20lecture%20slides/0140.jpg)
2. Compute budget for training LLMs
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/1d49615480e0f4446763d620078614ca63302834/screenshots%20of%20lecture%20slides/0141.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/1d49615480e0f4446763d620078614ca63302834/screenshots%20of%20lecture%20slides/0142.jpg)
3. Number of petaflop/s-days to pre-train various LLMs
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/9ce1bd104473ed38f64b03a36c7a0f6d062c42a5/screenshots%20of%20lecture%20slides/0143.jpg)
4. Compute budget vs. model performance
   - The lower test loss is, the better the performance
   - Compute resource constraints:
     * Hardware
     * Project timeline
     * Financial budget
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/613214e9860296f7b893ef5104c11ed202d21c5b/screenshots%20of%20lecture%20slides/0144.jpg)
5. Dataset size and model size vs. performance
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/232147b7bfe82a553feaa50b81c334b54d77cf23/screenshots%20of%20lecture%20slides/0146.jpg)
6. What's the ideal balance between these three quantities?
   - ChinChilla paper: The goal was to find the optimal number of parameters and volume of training data for a give compute budget.   
7. ChinChilla paper
   - Many of the 100 billion parameter LLM like GPT-3 may actually be over parameterized, they have more parameters than they need to achieve a good understanding of language and under trained so that they would benefit from seeing more trainig data.
   - Smaller models may able to achieve the same performance as much larger ones if they are trained on larger datasets.
   - The optimal training dataset size for a given model is about 20 times larger than the number of parameters in the model.
   - The compute optimal ChinChilla model outperforms non compute optimal models such as GPT-3 on a large range of downstream evaluation tasks.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/72ad0368cabcb74cfe8017e00bad0b6667a0cee6/screenshots%20of%20lecture%20slides/0148.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/72ad0368cabcb74cfe8017e00bad0b6667a0cee6/screenshots%20of%20lecture%20slides/0149.jpg)
8. With the results of the Chinchilla paper, teams have recently started to develop smaller modelsthat achieved similar, if not bettr results than larger models that were trained in a non-optimal way.
9. Bloomberg GPT, is trained in a compute optimal way follwing the Chinchilla and so achieves good performance with the size of 50 billiona parameters.

## Pre-training for domain adaption
1. If your target domain uses vocabulary and language structures that are not commonly used in day to day language. You may need to perform domain adaption to achieve good model performance.
2. Pretraining your model from scratch will result in better models for highly specilaized domains like laws, medicine, finance or science.
3. BloombergGPT is a LLM that has been pretrained for a specific domain - finance.
   - large Decoder-only language model
   - Result: achieves Best-in-class results on financial benchmarks. Also maintaing competitve performance on general purpose LLM benchmarks.
   - Data: 51% financial data  + 49% public data
5. The following two graphs compare a number of LLMs to scaling laws that have been discussed.
   - The number of parameters of BloombergGPT is fairly close to optimal.
   - The actual number of tokens used to pretrain BloombergGPT is below the recommended Chinchilla value for the available compute budget.
   - The smaller than optimal training dataset is due to limited availability financial domain data.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/56002b527db2444dbf44183c150fcd058bb3b105/screenshots%20of%20lecture%20slides/0156.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/786086a6f6bb540c410516d45b62d69f499e69ec/screenshots%20of%20lecture%20slides/BloombergGPT.png)






    
