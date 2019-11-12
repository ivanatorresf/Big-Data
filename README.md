### Big Data

* [What_is_Multilayer_perceptron_classifier?](#definition) 
* [Deep learning through multilayer perceptrons](#Deep_learning_through_multilayer_perceptrons)
* [Training a classifier is necessary:](#Training_a_classifier_is_necessary)  
* [Base and Activation Functions](#Base_and_Activation_Functions) 
* [Architecture](#Architecture) 
* [Operating stage](#Operating_stage)
* [Backpropagation algorithm](#Backpropagation_algorithm)
* [Lesson1Learning stage](#Learning_stage) 
* [Phases in the application of a multilayer perceptron](Application)
* [Limitations](#Limitations)
* [Video](#Video)


### What is Multilayer perceptron classifier?
Multilayer perceptron classifier (MLPC) is a classifier based on the feedforward artificial neural network. MLPC consists of multiple layers of nodes. Each layer is fully connected to the next layer in the network. Nodes in the input layer represent the input data. All other nodes map inputs to outputs by a linear combination of the inputs with the node’s weights w and bias b and applying an activation function.
![Figure 1: An Example of Multilayer Perceptron Architecture ](Imagen/multilayer.png)

#### Layer_Architecture_in_MLPC
* The input layer consists of neurons that accept the input values. The output from these neurons is same as the input predictors. Nodes in the input layer represent the input data. All other nodes map inputs to outputs by a linear combination of the inputs with the node’s weights w and bias b and applying an activation function. This can be written in matrix form for MLPC with K+1 layers as follows.
* Hidden layers are in between input and output layers. Typically, the number of hidden layers range from one to many. It is the central computation layer that has the functions that map the input to the output of a node. Nodes in the intermediate layers use the sigmoid (logistic) function, as follows.
* The output layer is the final layer of a neural network that returns the result back to the user environment. Based on the design of a neural network, it also signals the previous layers on how they have performed in learning the information and accordingly improved their functions. Nodes in the output layer use softmax.
### Deep_learning_through_multilayer_perceptrons
"Deep advance networks, also often called forward neural networks, or multilayer perceptrons (MLP), are the quintessential deep learning models."
-----

### Training_a_classifier_is_necessary:
#### To train a Spark based multilayer perceptron classifier, the following parameters must be set:
* Layer: Sets the value of param [[layers]].
* Tolerance of iteration: Set the convergence tolerance of iterations. Smaller value will lead to higher accuracy with the cost of more iterations. Default is 1E-4.
* Block size of the learning: Sets the value of param [[blockSize]], where, the default is 128.
* Seed size : Set the seed for weights initialization if weights are not set.
* Max iteration number: Set the maximum number of iterations. Default is 100.

Note that a smaller value of convergence tolerance will lead to higher accuracy with the cost of more iterations. The default block size parameter is 128 and the maximum number of iteration is set to be 100 as a default value. 

Moreover, adding just more hidden layers doesn't make it more accurate and productive. That can be done using two ways as follows:  

Adding more and more data since more data you supply to train the Deep Learning algorithm, better it becomes 

Furthermore, setting the optimal values of these parameters are a matter of hyperparameter tuning, therefore, I suggest you set these values accordingly and carefully. 

### Base_and_Activation_Functions
Base Function (Network Function)
The base function has two typical forms:
* Hyperplane linear function: The network value is a linear combination of the Inputs.
![Hyperplane linear function ](Imagen/funcion.png)
* Hyper spherical radial function: it is a nonlinear second order base function. The network value represents the distance to a specific reference pattern.
![Hyper spherical radial function](Imagen/funcion2.png)

Activation Function (Neuron Function)

The network value, expressed by the base function, u (w, x), is transformed by a non-linear activation function. The most common activation functions are the sigmoidal and Gaussian function:

![Sigmoidal function](Imagen/funcion3.png)

![Gaussian function](Imagen/funcion4.png)
### Architecture
![Architecture](Imagen/Architecture.png)

### Operating_stage

The neurons of this intermediate layer transform the received signals by applying an activation function thus providing an output value. This is transmitted through the Wkj weights towards the output layer, where applying the same operation as in the previous case, the neurons of this last layer provide the output of the network.
![Operating stage](Imagen/Architecture1.png)
![Operating stage](Imagen/Architecture2.png)

### Backpropagation_algorithm

It is considered an operation stage where, before the trained network, an input pattern is presented and this is transmitted through the successive layers of neurons until an output is obtained, and then a training or learning stage where the weights are modified of the network so that the output desired by the user matches the output obtained by the network.

![Backpropagation algorithm](Imagen/Architecture3.png)

![Backpropagation algorithm](Imagen/Architecture4.png)

### Learning_stage

In the learning stage, the objective is to minimize the error between the output obtained by the network and the output desired by the user before the presentation of a set of patterns called training group.

![Learning stage](Imagen/Architecture5.png)

![New values ​​in our original architecture](Imagen/Architecture6.png)

### Application
#### Phases in the application of a multilayer perceptron

A multilayer perceptron type network attempts to solve two types of problems:

- Prediction problems, which consists in the modification of a continuous variable of
output, from the presentation of a set of input predictive variables
(discrete and / or continuous).

- Classification problems, which consists in the assignment of the membership category
of a certain pattern from a set of input predictive variables (discrete and / or continuous).


### Limitations

* The Multilayer Perceptron does not extrapolate well, that is, if the network trains poorly or insufficiently, the outputs may be inaccurate.
* The existence of local minimums in the error function makes training considerably difficult, because once a minimum has been reached, training stops even if the set convergence rate has not been reached.
* When we fall to a local minimum without satisfying the percentage of error allowed, we can consider: changing the network topology (number of layers and number of neurons), starting the training with different initial weights, modifying the learning parameters, modifying the training set or present patterns in another order.

### Video
![Multilayer percuptron](Imagen/video.png)(https://youtu.be/u5GAVdLQyIg)
-----
 #CODE
            
            import org.apache.spark.ml.classification.MultilayerPerceptronClassifier
            import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
            // $example off$
            import org.apache.spark.sql.SparkSession

            /**
            * An example for Multilayer Perceptron Classification.
            */
            val spark = SparkSession.builder.appName("MultilayerPerceptronClassifierExample").getOrCreate()

                // $example on$
                // Load the data stored in LIBSVM format as a DataFrame.
                val data = spark.read.format("libsvm").load("sample_multiclass_classification_data.txt")

                // Split the data into train and test
                val splits = data.randomSplit(Array(0.6, 0.4), seed = 1234L)
                val train = splits(0)
                val test = splits(1)

                // specify layers for the neural network:
                // input layer of size 4 (features), two intermediate of size 5 and 4
                // and output of size 3 (classes)
                val layers = Array[Int](4, 5, 4, 3)

                // create the trainer and set its parameters
                val trainer = new MultilayerPerceptronClassifier().setLayers(layers).setBlockSize(128).setSeed(1234L).setMaxIter(100)

                // train the model
                val model = trainer.fit(train)

                // compute accuracy on the test set
                val result = model.transform(test)
                val predictionAndLabels = result.select("prediction", "label")
                val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")

                println(s"Test set accuracy = ${evaluator.evaluate(predictionAndLabels)}")
                // $example off$

 #OUTPUT
 ![output](Imagen/result.png)
#### Authors: 
#### Titule: Multilayer Perceptron Classifier
###### HERNANDEZ GARCIA RIGOBERTO           15212157
###### HERNÁNDEZ BOCANEGRA MIGUEL ANGEL     14211440
###### TORRES FLORES IVAN ADRIAN		    13210388
