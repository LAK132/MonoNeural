using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class Network
    {
        public static Random rand = new Random();
        
        /// <summary>
        /// Layer of neurons
        /// </summary
        public struct Layer
        {
            public List<Neuron> neuron;
            public uint outputs;
            public Layer(List<Neuron> parent, uint _outputs, uint count = 1, ActivateMode mode = ActivateMode.Sigmoid, double initMax = 0)
            {
                outputs = _outputs;
                neuron = new List<Neuron>();
                for (int i = 0; i < count; i++)
                {
                    neuron.Add(new Neuron(_outputs, mode));
                    neuron[i].link(parent, true);
                    neuron[i].init();
                }
            }

            public Layer(List<Neuron> other)
            {
                outputs = new uint();
                neuron = other;
                for (int i = 0; i < neuron.Count; i++)
                {
                    outputs += neuron[i].output.cols;
                }
            }
        }
        
        /// <summary>
        /// Group of layers used to make the network
        /// </summary
        public class Layers
        {
            /// <summary>
            /// Neuron used as an input to the network
            /// </summary>
            public List<Neuron> inputs;
            private ActivateMode outputMode;
            private bool outputSet = false;
            private uint outputs;
            private Neuron _output;
            public Neuron output
            {
                get
                {
                    if(!outputSet)
                    {
                        _output = new Neuron(outputs, outputMode);
                        _output.link(list[list.Count - 1].neuron, true);
                        _output.init();
                        outputSet = true;
                    }
                    return _output;
                }
            }
            public List<Layer> list = new List<Layer>();
            
            public Layers(List<Neuron> _inputs, uint _outputs, ActivateMode mode = ActivateMode.Sigmoid, List<Layer> layers = null)
            {
                if (layers != null) list = layers;
                inputs = _inputs;
                list.Add(new Layer(inputs));
                outputs = _outputs;
                outputMode = mode;
            }
            public Layers(uint _inputs, uint _outputs, ActivateMode mode = ActivateMode.Sigmoid, List<Layer> layers = null)
            {
                if (layers != null) list = layers;
                inputs = new List<Neuron>();
                inputs.Add(new Neuron(_inputs));
                list.Add(new Layer(inputs));
                outputs = _outputs;
                outputMode = mode;
            }
            public int Count
            {
                get { return list.Count; }
            }

            /// <summary>
            /// Creates a new layer of neurons and adds it to the end of the network
            /// </summary>
            public void create(uint outputs, uint count = 1, double initMax = 0, ActivateMode mode = ActivateMode.Sigmoid)
            {
                Layer prev = list[list.Count - 1];
                list.Add(new Layer(prev.neuron, outputs, count, mode, initMax));
                outputSet = false;
            }

            /// <summary>
            /// Adds the given layer to the end of the network
            /// </summary>
            public void add(Layer layer)
            {
                list.Add(layer);
                outputSet = false;
            }
            public Layer this[int key]
            {
                get { return list[key]; }
                //set { list[key] = value; }
            }
        }

        /// <summary>
        /// Set of input and output data
        /// </summary
        public struct dataSet
        {
            public Mat inputs;
            public Mat outputs;
            public double[] classes;
            public uint count;
            public Mat bias;
        }

        /// <summary>
        /// Stores the network state between epochs
        /// </summary
        public struct networkData
        {
            public uint inputCount;
            public uint outputCount;
            public Layers layers;
            public dataSet training;
            public dataSet validation;
            public dataSet test;
        }

        /// <summary>
        /// Converts the matrix output to classification output
        /// </summary>
        public static double[] outputToClasses(Mat A)
        {
            double[] T = new double[A.rows];
            for (uint row = 0; row < A.rows; row++)
            {
                uint maxCol = 0;
                for (uint col = 0; col < A.cols; col++)
                {
                    if (A[row, col] > A[row, maxCol])
                    {
                        maxCol = col;
                    }
                }
                T[row] = maxCol;
            }
            return T;
        }

        /// <summary>
        /// Converts the classification output to matrix output
        /// </summary
        public static Mat classesToOutput(uint[] A)
        {
            uint cols = 0;
            uint rows = (uint)A.Length;
            for (uint i = 0; i < A.Length; i++)
            {
                if (A[i] > cols)
                {
                    cols = A[i];
                }
            }
            Mat T = new Mat(rows, cols);
            for (uint row = 0; row < rows; row++)
            {
                for (uint col = 0; col < cols; col++)
                {
                    T[row, col] = (A[row] == col ? 1 : 0);
                }
            }
            return T;
        }

        public struct neOut
        {
            public double error;
            public double classError;
        }

        /// <summary>
        /// Calculates the raw output and classification error for the network
        /// </summary>
        public static neOut networkError(Layers layers, Mat tarOutputs, double[] tarClasses = null)
        {
            neOut rtn = new neOut();

            Mat output = layers.output.output;//think(inputs, layers).last;
            rtn.error = (~((tarOutputs - output) ^ 2)) / (layers[layers.Count - 1].outputs);

            if (tarClasses != null)
            {
                double[] classes = outputToClasses(output);
                uint errorCount = 0;
                for (int i = 0; i < classes.Length; i++)
                {
                    if (classes[i] != tarClasses[i])
                    {
                        errorCount++;
                    }
                }
                rtn.classError = errorCount / (double)layers.Count; // /sample_count
            }

            return rtn;
        }

        /// <summary>
        /// Trains the network for one epoch
        /// </summary>
        public static void train(ref networkData data, uint samples, double learnRate, double momentum, ref List<Tuple<float, Color>> graphPoints, uint graphWidth)
        {
            //Set up biases
            data.training.bias = new Mat(data.training.count, 1, 0);
            data.validation.bias = new Mat(data.validation.count, 1, 1);
            data.test.bias = new Mat(data.test.count, 1, 1);

            //Assign local input and output matricies
            Mat inputs = new Mat(data.training.inputs);//.concat(data.training.bias));
            Mat outputs = new Mat(data.training.outputs);

            //If more than one sample was received then take a random one
            if (inputs.rows > 1 || outputs.rows > 1)
            {
                uint sample = (uint)(rand.Next() / (Int32.MaxValue / Math.Min(Math.Min(samples, inputs.rows), outputs.rows)));
                inputs = inputs.row(sample);
                outputs = outputs.row(sample);
            }

            data.layers.inputs[0].output = inputs;

            //Check if the layers have been set up before, if not then create a default layer
            //if (data.layers.Count < 1) data.layers.create(data.outputCount, 1, 0.5);

            data.layers.output.update();
            Mat output = data.layers.output.output;
            data.layers.output.train(outputs, learnRate, momentum, true);//backProp(outputs, learnRate, momentum);

            //Do back propagation to adjust for error
            //var layers = backPropagation(ref data, inputs, outputs, learnRate, momentum);

            //Get the error for graphing
            var error = networkError(data.layers, outputs).error;

            //Graph the error
            graphPoints.RemoveAt(0);
            Tuple<float, Color> graphPoint = new Tuple<float, Color>((float)error, new Color(255, 255, 255));
            graphPoints.Add(graphPoint);
        }
    }
}
