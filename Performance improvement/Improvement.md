I will inject you a common list of parameters in machine learning and I want you to experiment with the ones that apply. Not for all of them. Only the ones that are commonly used in a task we have at our hand here. The style I want you to apply for this see pictures in the folder attached. There I did it for the learning rate, activation function and for the hidden sizes. Now, important. A) these pictures are just an example. B) You must choose the values to be tested based on the task we have at hand and the model that we want to build. In general use state of the art numbers that are commonly uses BUT always experiment and show how the numbers do 

The goal is to make out model better and the only way to do that is through experimentation and checking with success Metrix like accuracy or loss.  

[test best value for all that we see fit based on the success Metrix we have chosen, e.g.: accuracy and loss function]  
•	Learning rate (small, medium, large) 
•	Activation function (ReLu, tanh, sigmoid, Gelu) 
•	Hidden sizes 
•	Optimizer (SGD, Adam, RMS Prop, come up with more if suitable) 
•	Training Batch size (SGD, mini-batch, batch gradients, come up with more if suitable)
•	Number of epochs 
•	regularization techniques
•	Number of epochs (make this adjustable so I can change it myself) 

Architecture:
•	Channels 
•	Kernel size
•	Stride
•	Padding 

A methodology for fine-tuning transformers for classification tasks
1)	Pick Base pre-trained Architecture: Pick a base pre-trained architecture as a starting point for your fine-tuning. Example: bert-base-uncased is one such pre-trained model that can be loaded through Hugging Face Transformers Library
2)	Extract output from pre-training: How do you want to use the output from pre-training going into fine-tuning? a) Extract embedding from the first token, CLS b) Average embeddings of all tokens as a starting point (mean pooling).
3)	Add fine-tuning layers: Add fine-tuning layers on top of the pre-trained layers. Example, starting with the pooled embeddings, construct one or more dense layers (Feed-Forward NN style) to extract finer representations of the input. Add the output layer and its activation (typically softmax for classification tasks).
4)	Set training schedule, hyper-parameters, etc: Set up optimizer (e.g. ADAM), hyper-parameters, training schedule, etc for training

General rules:
•	Try to change the parameters always first before you change the architecture of the model
•	Stop training early if it shows signs of overfitting
•	If your model is overfitting (high training performance but low validation/testing performance)
o	Add dropout layers
o	Add regularization terms
o	Stop training early
o	Make network smaller (fewer layers or neurons)
