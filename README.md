# AppliedAI_IrisDataset_ELU_example

This repository serves as a research about ELU activation function implementation on iris dataset. It is supposed to use same neural network (architecture wise and hyperparameter value vise) but with different activation functions. Mainly, sigmoid and ELU functions are tested against each other.

# How to use

1. Clone https://github.com/arbbakbenny/NeurophFramework.git (this currently contains ELU function and neural net that uses it)
2. Setup dependencies by righ clicking on project and choosing **Properties**. In **Libraries** add references to
```
logback-core-1.1.2.jar
slf4j-api-1.7.5.jar
slf4j-nop-1.7.6.jar
neuroph-core-2.94.jar
neuroph-contrib-2.96.jar
```
3. Build using Netbeans by right clicking on project and choosing **Build with dependencies**
4. Run IrisClassification

# Possible problems:
If building NeurophFramework fails because of visrec-api. Solution is to clone and build https://github.com/JavaVisRec/visrec-api.git. After building it will stay in cache (on Windows OS - C:\Users\\{User}\\.m2\repository) and as such become available for building NeurophFramework.

Another problem that might pop up is failing tests, mainly following files:
```
neuroph/Contrib/src/test/java/org/neuroph/contrib/ConvolutionLayerTest.java
neuroph/Contrib/src/test/java/org/neuroph/contrib/ConvolutionNeuralNetworkTest.java
neuroph/Contrib/src/test/java/org/neuroph/contrib/FeatureMapTest.java
neuroph/Contrib/src/test/java/org/neuroph/contrib/PoolingLayerTest.java
neuroph/Core/src/test/java/org/neuroph/util/data/norm/ZeroMeanNormalizerTest.java
```
For the time being, just comment all the files.

# Description
This example tests multilayer perceptron network. One network uses ELU transfer function for its neurons while other uses sigmoid function. For a given set of hyperparameters it will give information about total iterations it took to minimise error and value of actual error. Usual result is that sigmoid function is better because it takes less iterations to reach better error value.

Whole example is based on lectures from 11th to 13th February 2018 for course Applied artificial intelligence at FON Belgrade, Serbia.
