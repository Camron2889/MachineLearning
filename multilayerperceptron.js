//requires: Perceptron

//class enclosure
(function() {
    "use strict";
    
    //constructor
    const MultiLayerPerceptron = function(/*numInputs, [numHidden1, numHidden2, ...] numOutputs*/) {
        this.numLayers = arguments.length - 1;
        this.layers = [];
        
        this.previousState = [];
        
        for (let i = 0; i < this.numLayers; i++) {
            const numInputs = arguments[i];
            const numOutputs = arguments[i + 1];
            this.layers[i] = new ml.Perceptron(numInputs, numOutputs);
            this.previousState[i] = this.layers[i].previousState;
        }

    };

    const proto = MultiLayerPerceptron.prototype;

    //public
    proto.setActivationFunction = function(args) { //accepts a string or a function
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].setActivationFunction(args);
        }
    };
    
    proto.calculate = function(inputs) {
        let cache = inputs;
        for (let i = 0; i < this.numLayers; i++) {
            cache = this.layers[i].calculate(cache);
        }
        return cache;
    };
    
    proto.calculateWithState = function(inputs) {
        let cache = inputs;
        for (let i = 0; i < this.numLayers; i++) {
            cache = this.layers[i].calculateWithState(cache);
        }
        return cache;
    };
    
    proto.randomize = function() {
        for (let i = 0; i < this.layers.length; i++) {
            this.layers[i].randomize();
        }
    };
    
    proto.toStaggeredArray = function() {
        const chromosome = [];
        for (let p = 0; p < this.numLayers; p++) {
            const layer = this.layers[p];
            const gene = layer.toStaggeredArray();
            for (let base = 0; base < gene.length; base++) {
                chromosome.push(gene[base]);
            };
        };
        
        return chromosome;
    };
    
    proto.fromStaggeredArray = function(chromosome) {
        let i = 0;
        for (let gene = 0; gene < this.numLayers; gene++) {
            const layer = this.layers[gene];
            const numBases = layer.numOutputs * layer.numInputs + layer.numOutputs;
            layer.fromStaggeredArray(chromosome.slice(i, i + numBases));
            
            i += numBases;
        };
    };

    //attach class to namespace
    ml.MultiLayerPerceptron = MultiLayerPerceptron;
})();