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
   - In the context of Distillation, the teacher model's output is often referred to as soft labels and the student model's predictions as soft predictions. In parallel, you train the student model to generate the correct predictions based on your ground truth training data. Here, you don't vary the temperature setting and instead use the standard softmax function. Distillation refers to the student model outputs as the hard predictions and hard labels. The loss between these two is the student loss. The combined distillation and student losses are used to update the weights of the student model via back propagation. The key benefit of distillation methods is that the smaller student model can be used for inference in deployment instead of the teacher model. In practice, distillation is not as effective for generative decoder models. It's typically more effective for encoder only models, such as Burt that have a lot of representation redundancy. Note that with Distillation, you're training a second, smaller model to use during inference. You aren't reducing the model size of the initial LLM in any way.









