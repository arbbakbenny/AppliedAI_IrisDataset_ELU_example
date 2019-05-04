/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package irisclassification;

import org.neuroph.contrib.transfer.ExponentialLinearUnit;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.transfer.Sigmoid;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.data.norm.Normalizer;
import org.neuroph.util.data.norm.RangeNormalizer;

/**
 *
 * @author student1
 */
public class IrisClassification {

    static int MAX_ITERATIONS = 20000;
    
    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        
        double bpLearningRate = 0.03d;
        
        int inputNum=4;
        int outputNum=3;
        String dataSetFile = "Iris-dataset-normalized.txt";
        
        DataSet irisData = DataSet.createFromFile(dataSetFile, inputNum, outputNum, "\t");
        Normalizer norm = new RangeNormalizer(0.1, 0.9);
        norm.normalize(irisData);        
        irisData.shuffle();
        DataSet[] trainTest = irisData.createTrainingAndTestSubsets(60, 40);
        
        
        MultiLayerPerceptron neuralNetELU = new MultilayerPerceptronELU(new ExponentialLinearUnit(0.1), inputNum, 16, outputNum);
        BackPropagation bpELU = neuralNetELU.getLearningRule();
        bpELU.setLearningRate(bpLearningRate);
        bpELU.setMaxIterations(MAX_ITERATIONS);
        bpELU.addListener(getListener("ELU"));
        neuralNetELU.learn(irisData); 
        
        
        //example of regular neural net with Sigmoid transfer
        MultiLayerPerceptron neuralNet = new MultiLayerPerceptron(inputNum, 16, outputNum);
        BackPropagation bp = neuralNet.getLearningRule();
//        bp.setLearningRate(bpLearningRate);
        bp.setLearningRate(0.6);
        bp.setMaxIterations(MAX_ITERATIONS);
        bp.addListener(getListener("Sigmoid"));
        neuralNet.learn(irisData);
        
        

        neuralNetELU = new MultilayerPerceptronELU(new ExponentialLinearUnit(0.1), inputNum, 16, 14, outputNum);
        bpELU = neuralNetELU.getLearningRule();
        bpELU.setLearningRate(bpLearningRate);
        bpELU.setMaxIterations(MAX_ITERATIONS);
        bpELU.addListener(getListener("ELU"));
        neuralNetELU.learn(irisData); 
        
        
        
        
        neuralNet = new MultiLayerPerceptron(inputNum, 16, 14, outputNum);
        bp = neuralNet.getLearningRule();
        bp.setLearningRate(0.6);
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
