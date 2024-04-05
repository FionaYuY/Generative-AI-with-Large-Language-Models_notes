# LLM-powered applications
## Model optimizations for deployment
1. Questions you'll have to consider
   - How your LLM will function in deployment
     + How fast do you need your model to generate completions
     + What compute budget do you have availabe
     + Are you willing to trade off model performance for improved inference speed or lower storage
   - Additional resources that you model may need
     + Do you intend for your model to interact with extenral data or other applications
     + How will you connenct your model to other resources
     + What will the intended application or API interface that your model will be consumed through look like?
2. LLM present inference challenges in terms of computing and storage requirements, as well as ensuring low latency for consuming applications.
   These challenges persist whether you're deploying an premises or to the cloud, and become even more of an issue when deploying to edge devices.
3. Reduce the size of the LLM
   - allow for quicker loading for the model, which reduces inference latency
   - The challenge is to reduce the size of the model while still maintaining model performance.
4. This lesson will cover three techniques that will work better
5. Distillation
   - Uses a larger model (the teacher model) to train a smaller model (the student model).
   - You then use the smaller model for inference to lower your storage and compute budget
