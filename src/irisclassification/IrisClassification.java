/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package irisclassification;

import java.util.ArrayList;
import java.util.List;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.data.norm.Normalizer;
import org.neuroph.util.data.norm.RangeNormalizer;

import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.contrib.transfer.ExponentialLinearUnit;
import org.neuroph.core.input.WeightedSum;
import org.neuroph.util.NeuronProperties;

/**
 *
 * @see https://arxiv.org/pdf/1804.02763.pdf
 */
public class IrisClassification {

    static int MAX_ITERATIONS = 20000;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        double bpLearningRate = 1d;
        double bpLearningRateELU = 1d;
        
        double eluAlpha = 0.5d;
        
        int inputNum=4;
        int outputNum=3;
        String dataSetFile = "Iris-dataset-normalized.txt";
        
        DataSet irisData = DataSet.createFromFile(dataSetFile, inputNum, outputNum, "\t");
        Normalizer norm = new RangeNormalizer(0.1, 0.9);
        norm.normalize(irisData);        
        irisData.shuffle();
        DataSet[] trainTest = irisData.createTrainingAndTestSubsets(60, 40);
        
        NeuronProperties neuronProperties = new NeuronProperties();
        neuronProperties.setProperty("useBias", true);
        neuronProperties.setProperty("transferFunction", ExponentialLinearUnit.class);
        neuronProperties.setProperty("transferFunctionAlpha", eluAlpha);
        neuronProperties.setProperty("inputFunction", WeightedSum.class);
        
        //example of multilayer network with ELU for neuron transfer function
        
        List<Integer> nnetParams = new ArrayList<>();
        nnetParams.add(inputNum);
        nnetParams.add(16);
        nnetParams.add(outputNum);
        
        MultiLayerPerceptron neuralNetELU = new MultiLayerPerceptron(nnetParams, neuronProperties);
        BackPropagation bpELU = neuralNetELU.getLearningRule();
        bpELU.setLearningRate(bpLearningRateELU);
        bpELU.setMaxIterations(MAX_ITERATIONS);
        bpELU.addListener(getListener("ELU"));
        neuralNetELU.learn(irisData); 
        
        //example of multilayer neural net with Sigmoid transfer function
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputNum, 16, outputNum);
        BackPropagation bp = neuralNet.getLearningRule();
        bp.setLearningRate(bpLearningRate);
        bp.setMaxIterations(MAX_ITERATIONS);
        bp.addListener(getListener("Sigmoid"));
        neuralNet.learn(irisData);
        
    }
    
    protected static LearningEventListener getListener(String activationFunction) {
        return new LearningEventListener() {
            
            public String lastMsg; 
            
            @Override
            public void handleLearningEvent(LearningEvent event) {
                BackPropagation bp = (BackPropagation) event.getSource();
                LearningEvent.Type eventName = event.getEventType();
                if (LearningEvent.Type.LEARNING_STOPPED.equals( eventName )) {
                    System.out.println(activationFunction + " activation function result - Iteration: "+bp.getCurrentIteration() + " Error: "+bp.getTotalNetworkError());
                }
            }
        };
    }
    
}
