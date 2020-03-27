//requires: MultiLayerPerceptron

//class enclosure
(function() {
    "use strict";
    
    //constructor
    const MlpView = function(mlp, parentElement = document.body, width = 320, height = 180) {
        this.mlp = mlp;
        this.parentElement = parentElement;
        
        //setup canvas
        this.canvas = document.createElement("canvas");
        this.parentElement.appendChild(this.canvas);
        this.canvas.setAttribute("style", "display: block; margin: auto; background-color: #000;");
        this.context = this.canvas.getContext("2d");
        
        //get largest layer size
        let biggest = mlp.numInputs;
        for (let i = 0; i < mlp.numLayers; i++) {
            const layerSize = mlp.layers[i].numOutputs;
            if (layerSize > biggest) {
                biggest = layerSize;
            }
        };
        this.biggestLayerSize = biggest;
        
        //create node position matrix
        this.nodePositions = [];
        for (let i = 0; i < mlp.numLayers + 1; i++) {
            this.nodePositions[i] = [];
        }
        
        //default settings
        this.nodeWidth = 10;
        this.absRange = 1;
        this.negColor = { r: 1, g: 0, b: 0, a: 1 };
        this.zeroColor = { r: 0, g: 0, b: 0, a: 0 };
        this.posColor = { r: 1, g: 1, b: 1, a: 1 };
        
        this.resize(width, height);
    };

    const proto = MlpView.prototype;

    //public
    proto.resize = function(width, height) {
        this.canvas.width = width;
        this.canvas.height = height;
        
        const mlp = this.mlp;
        
        const xSpacing = width / (mlp.numLayers + 2);
        const ySpacing = height / (this.biggestLayerSize + 1);
        
        let nodeX = xSpacing;
        let inodeY = (height - ySpacing * (mlp.numInputs - 1)) / 2;
        for (let j = 0; j < mlp.numInputs; j++) {
            this.nodePositions[0][j] = { x: nodeX, y: inodeY };
            inodeY += ySpacing;
        }
        nodeX += xSpacing;
        for (let i = 0; i < mlp.numLayers; i++) {
            const layer = mlp.layers[i];
            const colLength = layer.numOutputs;
            let nodeY = (height - ySpacing * (colLength - 1)) / 2;
            for (let j = 0; j < colLength; j++) {
                this.nodePositions[i + 1][j] = { x: nodeX, y: nodeY };
                nodeY += ySpacing;
            }
            nodeX += xSpacing;
        }
    };
    
    proto.draw = function() {
        const ctx = this.context;
        const width = this.canvas.width;
        const height = this.canvas.height;
        
        ctx.clearRect(0, 0, width, height);
        
        const nodes = this.nodePositions;
        
        //draw nodes
        for (let i = 0; i < nodes.length; i++) {
            let col = nodes[i];
            for (let j = 0; j < col.length; j++) {
                let value;
                if (i === 0) {
                    value = this.mlp.previousState[0].inputs[j];
                } else {
                    value = this.mlp.previousState[i - 1].outputs[j];
                };
                const color = this._gradientMap(value);
                ctx.fillStyle = `rgba(${color.r*255},${color.g*255},${color.b*255},${color.a})`;
                ctx.beginPath();
                const pos = col[j];
                ctx.arc(pos.x, pos.y, this.nodeWidth / 2, 0, Math.PI * 2);
                ctx.closePath();
                ctx.fill();
            }
        }
    };

    //private
    
    proto._colorLerp = function(color1, color2, ratio) {
        return {
            r: color1.r + (color2.r - color1.r) * ratio,
            g: color1.g + (color2.g - color1.g) * ratio,
            b: color1.b + (color2.b - color1.b) * ratio,
            a: color1.a + (color2.a - color1.a) * ratio,
        };
    };
    
    proto._gradientMap = function(x) {
        let color;
        
        if (x < 0) {
          color = this._colorLerp(this.negColor, this.zeroColor, (x + this.absRange) / this.absRange);
        } else if (x > 0) {
          color = this._colorLerp(this.zeroColor, this.posColor, x / this.absRange);
        } else {
          color = this.zeroColor;
        }
        
        return color;
    };

    //attach class to namespace
    ml.MlpView = MlpView;
})();