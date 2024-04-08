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


