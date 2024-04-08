# LLM-powered applications
## Model optimizations for deployment
1. Questions you'll have to consider to integrate your model into applications
   - How your LLM will function in deployment
     + How fast do you need your model to generate completions
     + What compute budget do you have available
     + Are you willing to trade off model performance for improved inference speed or lower storage
   - Additional resources that you model may need
     + Do you intend for your model to interact with extenral data or other applications
     + How will you connenct your model to other resources
     + What will the intended application or API interface that your model will be consumed through look like?
2. LLM present inference challenges in terms of computing and storage requirements, as well as ensuring low latency for consuming applications.
   These challenges persist whether you're deploying on premises or to the cloud, and become even more of an issue when deploying to edge devices.
3. Reduce the size of the LLM
   - allow for quicker loading for the model, which reduces inference latency
   - The challenge is to reduce the size of the model while still maintaining model performance.
4. This lesson will cover three techniques that will work better
   - Distillation
     + Uses a larger model (the teacher model) to train a smaller model (the student model).
     + You then use the smaller model for inference to lower your storage and compute budget
   - Quantization
     + Similar to quantization aware training, post training quantization transforms a model's weights to a lower precision representation, such as a 16- bit floating point or eight bit integer. This reduces the memory footprint of the model.
   - Pruning
     + Removes redundant model parameters that contribute little to the model's performance
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/bb8e0b6874aa93fe01e3cb873ac33bc62581f4de/Week3_screenshots/0074.jpg)
5. Model Distillation
   - a technique that focuses on having a larger teacher model train a smaller student model
   - The student model learns to statistically mimic the behavior of the teacher model, either just in the final prediction layer or in the model's hidden layers as well.
   - You start with your fine tune LLM as your teacher model and create a smaller LLM for your student model.
   - You freeze the teacher model's weights and use it to generate completions for your training data. At the same time, you generate completions for the training data using your student model.
   - The knowledge distillation between teacher and student model is achieved by minimizing a loss function called the distillation loss. To calculate this loss, distillation uses the probability distribution over tokens that is produced by the teacher model's softmax layer.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/bb8e0b6874aa93fe01e3cb873ac33bc62581f4de/Week3_screenshots/0076.jpg)
   - The teacher model is already fine tuned on the training data. So the probability distribution likely closely matches the ground truth data and won't have much variation in tokens. That's why Distillation applies a little trick adding a temperature parameter to the softmax function. A higher temperature increases the creativity of the language the model generates. With a temperature parameter greater than one, the probability distribution becomes broader and less strongly peaked. This softer distribution provides you with a set of tokens that are similar to the ground truth tokens.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/bb8e0b6874aa93fe01e3cb873ac33bc62581f4de/Week3_screenshots/0077.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/bb8e0b6874aa93fe01e3cb873ac33bc62581f4de/Week3_screenshots/0078.jpg)
   - In the context of Distillation, the teacher model's output is often referred to as soft labels and the student model's predictions as soft predictions. In parallel, you train the student model to generate the correct predictions based on your ground truth training data. Here, you don't vary the temperature setting and instead use the standard softmax function. Distillation refers to the student model outputs as the hard predictions and hard labels. The loss between these two is the student loss. The combined distillation and student losses are used to update the weights of the student model via back propagation. The key benefit of distillation methods is that the smaller student model can be used for inference in deployment instead of the teacher model.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/4892d7817c8760c571024eace9cf7a0fed787ce5/Week3_screenshots/0079.jpg)
i[image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/4892d7817c8760c571024eace9cf7a0fed787ce5/Week3_screenshots/0080.jpg)
   - In practice, distillation is not as effective for generative decoder models. It's typically more effective for encoder only models, such as Bert that have a lot of representation redundancy. Note that with Distillation, you're training a second, smaller model to use during inference. You aren't reducing the model size of the initial LLM in any way.
6. Post-Training Quantization (PTQ)
   - You were introduced to the second method, quantization, back in week one in the context of training, 'Specifically Quantization Aware Training' (QAT). However, after a model is trained, you can perform post training quantization (PTQ) to optimize it for deployment. 
   - PTQ transforms a model's weights to a lower precision representation, such as 16-bit floating point or 8-bit integer.
   - To reduce the model size and memory footprint, as well as the compute resources needed for model serving, quantization can be applied to just the model weights or to both weights and activation layers.
   - In general, quantization approaches that include the activations can have a higher impact on model performance. Quantization also requires an extra calibration step to statistically capture the dynamic range of the original parameter values.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/4892d7817c8760c571024eace9cf7a0fed787ce5/Week3_screenshots/0081.jpg)
   - As with other methods, there are tradeoffs because sometimes quantization results in a small percentage reduction in model evaluation metrics. However, that reduction can often be worth the cost savings and performance gains. 
7. Pruning
   - At a high level, the goal is to reduce model size for inference by eliminating weights that are not contributing much to overall model performance. These are the weights with values very close to or equal to zero.
   - Note that some pruning methods require full retraining of the model, while others fall into the category of parameter efficient fine tuning, such as LoRA. There are also methods that focus on post-training Pruning. In theory, this reduces the size of the model and improves performance. In practice, however, there may not be much impact on the size and performance if only a small percentage of the model weights are close to zero. 
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/4892d7817c8760c571024eace9cf7a0fed787ce5/Week3_screenshots/0082.jpg)
8. Quantization, Distillation and Pruning all aim to reduce model size to improve model performance during inference without impacting accuracy. 

## Generative AI Project Lifecycle Cheat Sheet
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/c48e2838178453a03f257e5546beee0b73fa29b3/Week3_screenshots/0083.jpg)
1. This cheat sheet provides some indication of the time and effort required for each phase of work.
2. Pre-training a large language model can be a huge effort. This stage is the most complex you'll face because of the model architecture decisions, the large amount of training data required, and the expertise needed. Remember though, that in general, you will start your development work with an existing foundation model. You'll probably be able to skip this stage.
3. If you're working with a foundation model, you'll likely start to assess the model's performance through prompt engineering, which requires less technical expertise, and no additional training of the model. If your model isn't performing as you need, you'll next think about prompt tuning and fine tuning. Depending on your use case, performance goals, and compute budget, the methods you'll try could range from full fine-tuning to parameter efficient fine tuning techniques like LoRA or prompt tuning. Some level of technical expertise is required for this work. But since fine-tuning can be very successful with a relatively small training dataset, this phase could potentially be completed in a single day.
4. Aligning your model using reinforcement learning from human feedback can be done quickly, once you have your train reward model. You'll likely see if you can use an existing reward model for this work. However, if you have to train a reward model from scratch, it could take a long time because of the effort involved to gather human feedback.
5. Finally, optimization techniques you learned about in the last video, typically fall in the middle in terms of complexity and effort, but can proceed quite quickly assuming the changes to the model don't impact performance too much. After working through all of these steps, you have hopefully trained in tuned a gray LLM that is working well for your specific use case, and is optimized for deployment.

## Using the LLM in applications
1. There are some broader challenges with LLM that can't be solved by training alone
   - One ussue is that the internal knoweldge held by a model cuts off at the moment of pretraining.
   - Struggle with complex math
     + Note that LLMs do not carry out mathematical operations. They are still just trying to predict the next best token based on their training.
   - Their tendency to generate text even when they don't know the answer to a problem (hallucination)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0085.jpg)
2.  You can help your LLM overcome these issues by connecting to external data sources and applications. You'll be able to connect your LLM to these external components and fully integrate everything for deployment within your application. Your application must manage the passing of user input to the LLM and the return of completions. This is often done through some type of 'Orchestration library'. This layer can enable some powerful technologies that augment and enhance the performance of the LLM at runtime. By providing access to external data sources or connecting to existing APIs of other applications. One implementation example is Langchain.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0088.jpg)
3. Retrieval Augmented Generation (RAG)
   - is a framework for building LLM powered systems that make use of external data sources and applications to overcome some of the limitations of these models
   - RAG is a great way to overcome the knowledge cutoff issue and help the model update its understanding of the world. While you could retrain the model on new data, this would quickly become very expensive. And require repeated retraining to regularly update the model with new knowledge. A more flexible and less expensive way to overcome knowledge cutoffs is to give your model access to additional external data at inference time.  
   - RAG is useful in any case where you want the language model to have access to data that it may not have seen. This could be new information documents not included in the original training data, or proprietary knowledge stored in your organization's private databases. Providing your model with external information, can improve both the relevance and accuracy of its completions.
   -  Retrieval augmented generation isn't a specific set of technologies, but rather a framework for providing LLMs access to data they did not see during training. A number of different implementations exist, and the one you choose will depend on the details of your task and the format of the data you have to work with. 
4. Here you'll walk through the implementation discussed in one of the earliest papers on RAG by researchers at Facebook, originally published in 2020.
   - At the heart of this implementation is a model component called the Retriever, which consists of a query encoder and an external data source.
   - The encoder takes the user's input prompt and encodes it into a form that can be used to query the data source. In the Facebook paper, the external data is a vector store. But it could instead be a SQL database, CSV files, or other data storage format.
   - These two components are trained together to find documents within the external data that are most relevant to the input query. The Retriever returns the best single or group of documents from the data source and combines the new information with the original user query.
   - The new expanded prompt is then passed to the language model, which generates a completion that makes use of the data. 
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0092.jpg)
5. An example: Imagine you are a lawyer using a LLM
   - A RAG architecture can help you ask questions of a corpus of documents
   - Here you ask the model about the plaintiff named in a specific case number. The prompt is passed to the query encoder, which encodes the data in the same format as the external documents. And then searches for a relevant entry in the corpus of documents.
   - Having found a piece of text that contains the requested information, the Retriever then combines the new text with the original prompt. The expanded prompt that now contains information about the specific case of interest is then passed to the LLM.
   - The model uses the information in the context of the prompt to generate a completion that contains the correct answer. The use case you have seen here is quite simple and only returns a single piece of information that could be found by other means. But imagine the power of Rag to be able to generate summaries of filings or identify specific people, places and organizations within the full corpus of the legal documents.
   - Allowing the model to access information contained in this external data set greatly increases its utility for this specific use case.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0093.jpg)
6. In addition to overcoming knowledge cutoffs, rag also helps you avoid the problem of the model hallucinating when it doesn't know the answer.
7. RAG architectures can be used to integrate multiple types of external information sources. You can augment LLM with access to local documents, including private wikis and expert systems. Rag can also enable access to the Internet to extract information posted on web pages, for example, Wikipedia. By encoding the user input prompt as a SQL query, RAG can also interact with databases. Another important data storage strategy is a Vector Store, which contains vector representations of text. This is a particularly useful data format for language models, since internally they work with vector representations of language to generate text. Vector stores enable a fast and efficient kind of relevant search based on similarity.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0095.jpg)
8. Note that implementing RAG is a little more complicated than simply adding text into the large language model. There are a couple of key considerations to be aware of.
   - Starting with the size of the context window. Most text sources are too long to fit into the limited context window of the model, which is still at most just a few thousand tokens. Instead, the external data sources are chopped up into many chunks, each of which will fit in the context window. Packages like Langchain can handle this work for you.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0096.jpg)
   - Second, the data must be available in a format that allows for easy retrieval of the most relevant text. Recall that large language models don't work directly with text, but instead create vector representations of each token in an embedding space. These embedding vectors allow the LLM to identify semantically related words through measures such as cosine similarity.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0097.jpg)
   - Rag methods take the small chunks of external data and process them through the LLM, to create embedding vectors for each. These new representations of the data can be stored in structures called vector stores, which allow for fast searching of datasets and efficient identification of semantically related text.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0098.jpg)
   - Vector databases are a particular implementation of a vector store where each vector is also identified by a key. This can allow, for instance, the text generated by RAG to also include a citation for the document from which it was received.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0099.jpg)
9. By providing up to date relevant information and avoiding hallucinations, you can greatly improve the experience of using your application for your users.

## Interacting with edxternal applications
1. During this walkthrough of one customer's interaction with ShopBot, you'll take a look at the integrations that you'd need to allow the app to process a return requests from end to end.
   - In this conversation, the customer has expressed that they want to return some genes that they purchased. ShopBot responds by asking for the order number, which the customer then provides. ShopBot then looks up the order number in the transaction database. One way it could do this is by using a rag implementation of the kind you saw earlier in the previous video.
   - In this case here, you would likely be retrieving data through a SQL query to a back-end order database rather than retrieving data from a corpus of documents. Once ShopBot has retrieved the customers order, the next step is to confirm the items that will be returned. The bot ask the customer if they'd like to return anything other than the genes.
   - After the user states their answer, the bot initiates a request to the company's shipping partner for a return label. The body uses the shippers Python API to request the label ShopBot is going to email the shipping label to the customer. It also asks them to confirm their email address. The customer responds with their email address and the bot includes this information in the API call to the shipper. Once the API request is completed, the bot lets the customer know that the label has been sent by email, and the conversation comes to an end.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0101.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0102.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0103.jpg)
2. In general, connecting LLMs to external applications allows the model to interact with the broader world, extending their utility beyond language tasks.
   - As the shop bot example showed, LLMs can be used to trigger actions when given the ability to interact with APIs.
   - LLMs can also connect to other programming resources. For example, a Python interpreter that can enable models to incorporate accurate calculations into their outputs.
   - It's important to note that prompts and completions are at the very heart of these workflows. The actions that the app will take in response to user requests will be determined by the LLM, which serves as the application's reasoning engine.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0104.jpg)
3. In order to trigger actions, the completions generated by the LLM must contain certain important information.
   - First, the model needs to be able to generate a set of instructions so that the application knows what actions to take. These instructions need to be understandable and correspond to allowed actions. In the ShopBot example for instance, the important steps were; checking the order ID, requesting a shipping label, verifying the user email, and emailing the user the label.
   - Second, the completion needs to be formatted in a way that the broader application can understand. This could be as simple as a specific sentence structure or as complex as writing a script in Python or generating a SQL command. For example, here is a SQL query that would determine whether an order is present in the database of all orders.
   - Lastly, the model may need to collect information that allows it to validate an action. For example, in the ShopBot conversation, the application needed to verify the email address the customer used to make the original order. Any information that is required for validation needs to be obtained from the user and contained in the completion so it can be passed through to the application.
   - Structuring the prompts in the correct way is important for all of these tasks and can make a huge difference in the quality of a plan generated or the adherence to a desired output format specification.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/3281ab5a11559ade7f88e3b7193704ef43bfe7f9/Week3_screenshots/0105.jpg)
​
## Helping LLMs reason and plan with chain-of-thought
1. Complex reasoning can be challenging for LLMs, especially for problems involve multiple steps or methematics. These problems exist even in large models that show good performance at many other tasks.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/6991ee193336eba79dc8d129038ade2d6c9debd3/Week3_screenshots/0107.jpg)
2. Researchers have been exploring ways to improve the performance of large language models on reasoning tasks, like the one you just saw. One strategy that has demonstrated some success is prompting the model to think more like a human, by breaking the problem down into steps.
3. The task here is to calculate how many tennis balls Roger has after buying some new ones. One way that a human might tackle this problem is as follows. Begin by determining the number of tennis balls Roger has at the start. Then note that Roger buys two cans of tennis balls. Each can contains three balls, so he has a total of six new tennis balls. Next, add these 6 new balls to the original 5, for a total of 11 balls. Then finish by stating the answer.
4. These intermediate calculations form the reasoning steps that a human might take, and the full sequence of steps illustrates the chain of thought that went into solving the problem. Asking the model to mimic this behavior is known as 'chain of thought prompting'. 
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/6991ee193336eba79dc8d129038ade2d6c9debd3/Week3_screenshots/0108.jpg)
5. Chain of thought prompting
   - It works by including a series of intermediate reasoning steps into any examples that you use for one or few-shot inference. By structuring the examples in this way, you're essentially teaching the model how to reason through the task to reach a solution.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/6991ee193336eba79dc8d129038ade2d6c9debd3/Week3_screenshots/0109.jpg)
   - Notice that the model has now produced a more robust and transparent response that explains its reasoning steps, following a similar structure as the one-shot example. Thinking through the problem has helped the model come to the correct answer. One thing to note is that while the input prompt is shown here in a condensed format to save space, the entire prompt is actually included in the output. You can use chain of thought prompting to help LLMs improve their reasoning of other types of problems too, in addition to arithmetic.
   - Chain of thought prompting is a powerful technique that improves the ability of your model to reason through problems. While this can greatly improve the performance of your model, the limited math skills of LLMs can still cause problems if your task requires accurate calculations, like totaling sales on an e-commerce site, calculating tax, or applying a discount.

## Program-aided language models (PAL)
1. While you can try using chain of thought prompting to overcome some part of the limitation of mathematical operation of LLMs, it may still get the individual math operations wrong even if the modle correctly reasons through a problem, especially with larger numbers or complex operations.
2. Remember, the model isn't actually doing any real math. It is simply trying to predict the most probable tokens that complete the prompt. 
3. You can overcome this limitation by allowing your model to interact with external applications that are good at math, like a Python interpreter. One interesting framework for augmenting LLMs in this way is called program-aided language models (PAL).
4. PAL
   - PAL pairs an LLM with an external code interpreter to carry out calculations.
   - The method makes use of chain of thought prompting to generate executable Python scripts. The scripts that the model generates are passed to an interpreter to execute. The image on the right here is taken from the paper and show some example prompts and completions. The strategy behind PAL is to have the LLM generate completions where reasoning steps are accompanied by computer code. This code is then passed to an interpreter to carry out the calculations necessary to solve the problem. You specify the output format for the model by including examples for one or few short inference in the prompt. 
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/5f726da2dc88b94e74e6a48352ecf3415a9e3aef/Week3_screenshots/0113.jpg)
5. Example
   - You can see the reasoning steps written out in words on the lines highlighted in blue. What differs from the prompts you saw before is the inclusion of lines of Python code shown in pink. These lines translate any reasoning steps that involve calculations into code. Variables are declared based on the text in each reasoning step. Their values are assigned either directly, as in the first line of code here, or as calculations using numbers present in the reasoning text as you see in the second Python line.
   - The model can also work with variables it creates in other steps, as you see in the third line. Note that the text of each reasoning step begins with a pound(#) sign, so that the line can be skipped as a comment by the Python interpreter. The prompt here ends with the new problem to be solved. In this case, the objective is to determine how many loaves of bread a bakery has left after a day of sales and after some loaves are returned from a grocery store partner.
   - On the right, you can see the completion generated by the LLM. Again, the chain of thought reasoning steps are shown in blue and the Python code is shown in pink. As you can see, the model creates a number of variables to track the loaves baked, the loaves sold in each part of the day, and the loaves returned by the grocery store. The answer is then calculated by carrying out arithmetic operations on these variables. The model correctly identifies whether terms should be added or subtracted to reach the correct total. 
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/5f726da2dc88b94e74e6a48352ecf3415a9e3aef/Week3_screenshots/0115.jpg)
6. How PAL framework enables an LLM to interact with an external intepreter
   - To prepare for inference with PAL, you'll format your prompt to contain one or more examples. Each example should contain a question followed by reasoning steps in lines of Python code that solve the problem.
   - Next, you will append the new question that you'd like to answer to the prompt template. Your resulting PAL formatted prompt now contains both the example and the problem to solve.
   - Next, you'll pass this combined prompt to your LLM, which then generates a completion that is in the form of a Python script having learned how to format the output based on the example in the prompt. You can now hand off the script to a Python interpreter, which you'll use to run the code and generate an answer.
   - For the bakery example script, the answer is 74. You'll now append the text containing the answer, which you know is accurate because the calculation was carried out in Python to the PAL formatted prompt you started with.
   - By this point you have a prompt that includes the correct answer in context. Now when you pass the updated prompt to the LLM, it generates a completion that contains the correct answer. Given the relatively simple math in the bakery bread problem, it's likely that the model may have gotten the answer correct just with chain of thought prompting. But for more complex math, including arithmetic with large numbers, trigonometry or calculus, PAL is a powerful technique that allows you to ensure that any calculations done by your application are accurate and reliable.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/5f726da2dc88b94e74e6a48352ecf3415a9e3aef/Week3_screenshots/0116.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/5f726da2dc88b94e74e6a48352ecf3415a9e3aef/Week3_screenshots/0117.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/5f726da2dc88b94e74e6a48352ecf3415a9e3aef/Week3_screenshots/0118.jpg)
7. You might be wondering how to automate this process so that you don't have to pass information back and forth between the LLM, and the interpreter by hand. This is where the orchestrator that you saw earlier comes in. The orchestrator shown here as the yellow box is a technical component that can manage the flow of information and the initiation of calls to external data sources or applications. It can also decide what actions to take based on the information contained in the output of the LLM.
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/5f726da2dc88b94e74e6a48352ecf3415a9e3aef/Week3_screenshots/0119.jpg)
![image](https://github.com/FionaYuY/Generative-AI-with-Large-Language-Models_notes/blob/5f726da2dc88b94e74e6a48352ecf3415a9e3aef/Week3_screenshots/0120.jpg)
9. Remember, the LLM is your application's reasoning engine. Ultimately, it creates the plan that the orchestrator will interpret and execute. In PAL there's only one action to be carried out, the execution of Python code. The LLM doesn't really have to decide to run the code, it just has to write the script which the orchestrator then passes to the external interpreter to run. However, most real-world applications are likely to be more complicated than the simple PAL architecture. Your use case may require interactions with several external data sources. As you saw in the shop bot example, you may need to manage multiple decision points, validation actions, and calls to external applications. 

## ReAct: combining reasoning and action
1. 











