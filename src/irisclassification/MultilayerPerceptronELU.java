/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package irisclassification;

import java.util.ArrayList;
import java.util.List;
import org.neuroph.nnet.MultiLayerPerceptron;

import org.neuroph.util.NeuronProperties;
import org.neuroph.util.TransferFunctionType;

import org.neuroph.contrib.transfer.ExponentialLinearUnit;
import org.neuroph.core.Layer;
import org.neuroph.core.input.WeightedSum;
import org.neuroph.core.transfer.Linear;
import org.neuroph.nnet.comp.neuron.BiasNeuron;
import org.neuroph.nnet.comp.neuron.InputNeuron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.ConnectionFactory;
import org.neuroph.util.LayerFactory;
import org.neuroph.util.NeuralNetworkFactory;
import org.neuroph.util.NeuralNetworkType;
import org.neuroph.util.random.RangeRandomizer;

import org.neuroph.core.Neuron;

public class MultilayerPerceptronELU extends MultiLayerPerceptron {
    
    public MultilayerPerceptronELU(List<Integer> neuronsInLayers) {
        super(neuronsInLayers);
    }
    
    public MultilayerPerceptronELU(ExponentialLinearUnit elu, int... neuronsInLayers) {
        super(neuronsInLayers);    
        
        for (int i = 1; i < this.getLayersCount(); i++) {
            for( Object neuron : this.getLayerAt(i).getNeurons() ) {
                ((Neuron) neuron).setTransferFunction(elu);
            }
        }
    }
    
}
