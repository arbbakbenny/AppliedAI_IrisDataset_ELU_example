/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package irisclassification;

import org.neuroph.core.data.DataSet;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.transfer.TransferFunction;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;
import org.neuroph.util.data.norm.Normalizer;
import org.neuroph.util.data.norm.RangeNormalizer;

/**
 *
 * @author student1
 */
public class IrisClassification {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        int inputNum=4;
        int outputNum=3;
        String dataSetFile = "Iris-dataset-normalized.txt";
        
        DataSet irisData = DataSet.createFromFile(dataSetFile, inputNum, outputNum, "\t");
        Normalizer norm = new RangeNormalizer(0.1, 0.9);
        norm.normalize(irisData);        
        irisData.shuffle();
        DataSet[] trainTest = irisData.createTrainingAndTestSubsets(60, 40);
        
        MultiLayerPerceptron neuralNet = new MultilayerPerceptronELU(inputNum, 16, outputNum);
        BackPropagation bp = neuralNet.getLearningRule();
        bp.setLearningRate(0.6);
        
        bp.addListener(new LearningEventListener() {
            @Override
            public void handleLearningEvent(LearningEvent event) {
                BackPropagation bp = (BackPropagation) event.getSource();
                System.out.println("Iteration: "+bp.getCurrentIteration() + " Error: "+bp.getTotalNetworkError());
            }
        });
        
        neuralNet.learn(irisData); 
        
        System.out.println("Done!");
    }
    
}
