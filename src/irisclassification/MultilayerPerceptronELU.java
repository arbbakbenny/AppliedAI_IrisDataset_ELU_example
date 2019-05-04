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
import org.neuroph.core.transfer.TransferFunction;

public class MultilayerPerceptronELU extends MultiLayerPerceptron {
    
    public MultilayerPerceptronELU(List<Integer> neuronsInLayers) {
        super(neuronsInLayers);
    }
    
    public MultilayerPerceptronELU(TransferFunction elu, int... neuronsInLayers) {
        super(neuronsInLayers);    
        
        for (int i = 1; i < this.getLayersCount(); i++) {
            for( Object neuron : this.getLayerAt(i).getNeurons() ) {
                ((Neuron) neuron).setTransferFunction(elu);
            }
        }
    }
    
    /**
     * Creates MultiLayerPerceptron Network architecture - fully connected
     * feed forward with specified number of neurons in each layer
     *
     * @param neuronsInLayers  collection of neuron numbers in getLayersIterator
     * @param neuronProperties neuron properties
     */
    private void createNetwork(List<Integer> neuronsInLayers, NeuronProperties neuronProperties) {

        // set network type
        this.setNetworkType(NeuralNetworkType.MULTI_LAYER_PERCEPTRON);

        // create input layer
        NeuronProperties inputNeuronProperties = new NeuronProperties(InputNeuron.class, Linear.class);
        Layer layer = LayerFactory.createLayer(neuronsInLayers.get(0), inputNeuronProperties);

        boolean useBias = true; // use bias neurons by default
        if (neuronProperties.hasProperty("useBias")) {
            useBias = (Boolean) neuronProperties.getProperty("useBias");
        }

        if (useBias) {
            layer.addNeuron(new BiasNeuron());
        }

        this.addLayer(layer);

        // create layers
        Layer prevLayer = layer;

        //for(Integer neuronsNum : neuronsInLayers)
        for (int layerIdx = 1; layerIdx < neuronsInLayers.size(); layerIdx++) {
            Integer neuronsNum = neuronsInLayers.get(layerIdx);
            // createLayer layer
            layer = LayerFactory.createLayer(neuronsNum, neuronProperties);

            if (useBias && (layerIdx < (neuronsInLayers.size() - 1))) {
                layer.addNeuron(new BiasNeuron());
            }

            // add created layer to network
            this.addLayer(layer);
            // createLayer full connectivity between previous and this layer
            if (prevLayer != null) {
                ConnectionFactory.fullConnect(prevLayer, layer);
            }

            prevLayer = layer;
        }

        // set input and output cells for network
        NeuralNetworkFactory.setDefaultIO(this);

        // set learnng rule
//        this.setLearningRule(new BackPropagation());
        this.setLearningRule(new MomentumBackpropagation());
        // this.setLearningRule(new DynamicBackPropagation());

        this.randomizeWeights(new RangeRandomizer(-0.7, 0.7));

    }
    
}
