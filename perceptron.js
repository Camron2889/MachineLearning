//namespace
this.ml = this.ml || {};

//class enclosure
(function() {
    "use strict";
    
    //constructor
    const Perceptron = function(numInputs, numOutputs) {
        this.numInputs = numInputs;
        this.numOutputs = numOutputs;
        this.numWeights = numInputs * numOutputs;
        
        this.weights = [];
        for (let i = 0; i < this.numWeights; i++) {
            this.weights[i] = 0;
        }
        
        this.biases = [];
        for (let i = 0; i < numOutputs; i++) {
            this.biases[i] = 0;
        }
        this.setActivationFunction("tanh");

    };

    const proto = Perceptron.prototype;

    //public
    proto.setActivationFunction = function(args) { //accepts a string or a function
        const t = typeof args;
        if (t === "string") {
            switch (args) {
                case "sigmoid":
                    this._activationFunction = this._sigmoid;
                    break;
                case "tanh":
                    this._activationFunction = Math.tanh;
                    break;
            }
        } else if (t === "function") {
            this._activationFunction = args;
        }
    };
    
    proto.calculate = function(inputs) {
        const numOut = this.numOutputs;
        const numIn = this.numInputs;
        const results = [];
        for (let opt = 0; opt < numOut; opt++) {
            let sum = 0;
            for (let ipt = 0; ipt < numIn; ipt++) {
                const wt = opt * numIn + ipt;
                sum += this.weights[wt] * inputs[ipt];
            }
            sum += this.biases[opt];
            results[opt] = this._activationFunction(sum);
        }
        return results;
    };
    
    proto.randomize = function() {
        for (let i = 0; i < this.numWeights; i++) {
            this.weights[i] = Math.random() * 2 - 1;
        }
        for (let i = 0; i < this.numOutputs; i++) {
            this.biases[i] = Math.random() * 2 - 1;
        }
    };

    //private
    proto._sigmoid = function(x) {
        return 1 / (1 + Math.pow(Math.E, -x));
    };
    
    proto._sigmoidDerivative = function(x) {
        const y = this._sigmoid(x);
        return y * (1 - y);
    };
    
    proto._tanhDerivative = function(x) {
        const y = Math.tanh(x);
        return 1 - (y * y);
    };

    //attach class to namespace
    ml.Perceptron = Perceptron;
})();