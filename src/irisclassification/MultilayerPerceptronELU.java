/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package irisclassification;

import java.util.List;
import org.neuroph.nnet.MultiLayerPerceptron;

import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

import org.neuroph.contrib.transfer.ExponentialLinearUnit;

public class MultilayerPerceptronELU extends MultiLayerPerceptron {
    
    public MultilayerPerceptronELU(List<Integer> neuronsInLayers) {
        super(neuronsInLayers);
    }

    public MultilayerPerceptronELU(int... neuronsInLayers) {
        super(neuronsInLayers);
    }

    public MultilayerPerceptronELU(TransferFunctionType transferFunctionType, int... neuronsInLayers) {
        super(transferFunctionType, neuronsInLayers);
    }

    public MultilayerPerceptronELU(List<Integer> neuronsInLayers, TransferFunctionType transferFunctionType) {
        super(neuronsInLayers, transferFunctionType);
    }

    public MultilayerPerceptronELU(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {
        super(neuronsInLayers, neuronProperties);
    }
    
}
