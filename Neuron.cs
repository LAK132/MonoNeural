using Microsoft.Xna.Framework;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    public enum ActivateMode
    {
        Identity,
        Sigmoid,
        TanH
    }

    class Neuron
    {
        public class Child
        {
            public Neuron neuron;
            public Mat backDelta;
            public Child(Neuron n, Mat bd)
            {
                neuron = n;
                backDelta = bd;
            }
        }

        public static Random rand = new Random();
        public ActivateMode mode;
        public List<Neuron> parents = new List<Neuron>();
        public List<Child> children = new List<Child>();
        private uint outputWidth;
        private bool trained = false;
        private bool initialised = false;
        private List<Mat> inputs = new List<Mat>();
        private Mat weights;
        private List<Mat> backDelta = new List<Mat>(); //Changed by child, linked to the children tuple
        private Mat delta;


        private Mat _prevDelta;
        private Mat prevDelta
        {
            get { return _prevDelta; }
            set { _prevDelta.clone(value); } //= operator would destroy the reference to the correct object
        }
        private Mat _output;
        public Mat output
        {
            get { return _output; }
            set { _output.clone(value); } //= operator would destroy the reference to the correct object
        }

        public Neuron(uint outputCount, ActivateMode activateMode = ActivateMode.Sigmoid)
        {
            outputWidth = outputCount;
            _output = new Mat(1, outputWidth, 0);
            delta = new Mat(1, outputWidth, 0);
            _prevDelta = new Mat(delta);
            mode = activateMode;
        }

        /// <summary>
        /// Finalise inputs and create weights. DO NOT EXECUTE IF NO INPUTS HAVE BEEN ADDED
        /// </summary>
        public void init(double weightsInitMax = 0)
        {
            if (weights == null)
                weights = weightInit(input.cols, outputWidth, weightsInitMax);
            else
                weights.extend(input.cols, outputWidth, weightsInitMax);
            initialised = true;
        }

        /// <summary>
        /// Link this neuron to another 
        /// </summary>
        public void link(Neuron other, bool inputNeuron)
        {
            if (inputNeuron)
            {
                Neuron self = this;
                Mat bdelta = new Mat(1, other.outputWidth, 0);
                other.children.Add(new Child(self, bdelta));
                backDelta.Add(bdelta);
                inputs.Add(other.output);
                parents.Add(other);
            }
            else //this neuron outputs to 'other' neuron
            {
                Mat bdelta = new Mat(1, outputWidth, 0);
                children.Add(new Child(other, bdelta));
                other.backDelta.Add(bdelta);
                other.inputs.Add(output);
                other.parents.Add(this);
            }
        }

        /// <summary>
        /// Link this neuron to a set of neurons
        /// </summary>
        public void link(List<Neuron> other, bool inputNeuron)
        {
            for (int i = 0; i < other.Count; i++)
            {
                Neuron n = other[i];
                link(n, inputNeuron);
            }
        }

        /// <summary>
        /// Returns the grouped outputs of the parent neurons
        /// </summary>
        public Mat input
        {
            get
            {
                Mat inp = new Mat(1, 0);
                for (int i = 0; i < inputs.Count; i++)
                {
                    inp = inp.concat(inputs[i]);
                }
                return inp;
            }
        }

        /// <summary>
        /// Initialises a matrix with random values between max and -max
        /// </summary>
        public static Mat weightInit(uint rows, uint cols, double max)
        {
            Mat T = new Mat(rows, cols);
            for (uint row = 0; row < T.rows; row++)
            {
                for (uint col = 0; col < T.cols; col++)
                {
                    T[row, col] = ((2 * max) * (1 - rand.NextDouble())) - max;
                }
            }
            return T;
        }

        /// <summary>
        /// Network "activation fuction"
        /// </summary>
        private static Mat activate(Mat X, ActivateMode mode = ActivateMode.Sigmoid, bool deriv = false)
        {
            Mat Y = new Mat(X.rows, X.cols);
            for (uint row = 0; row < Y.rows; row++)
            {
                for (uint col = 0; col < Y.cols; col++)
                {
                    switch (mode)
                    {
                        case ActivateMode.Identity:
                            Y[row, col] = (deriv ? 1 : X[row, col]);
                            break;
                        case ActivateMode.Sigmoid:
                            double sig = 1 / (1 + Math.Exp(-X[row, col]));
                            Y[row, col] = (deriv ? sig * (1 - sig) : sig);
                            break;
                        case ActivateMode.TanH:
                            double tanh = (2 / (1 + Math.Exp(-2 * X[row, col]))) - 1;
                            Y[row, col] = (deriv ? 1 - (tanh * tanh) : tanh);
                            break;
                    }
                }
            }
            return Y;
        }

        /// <summary>
        /// Updates this neurons output
        /// If flowDown = true then all parent (and grandparent) neurones are updated aswell
        /// </summary>
        public void update(bool flowDown = false)
        {
            if (initialised)
            {
                trained = false;

                //Call update for all parents
                if (flowDown)
                {
                    for (int i = 0; i < parents.Count; i++)
                    {
                        parents[i].update(flowDown);
                    }
                }

                //Update output
                output = activate(input * weights, mode);
            }
        }

        /// <summary>
        /// If this is the final neuron in the networ, find the network error based on the targeted outputs
        /// If this isn't the final neuron then find the error based on the 
        /// </summary>
        public void train(Mat target, double learnRate = 1.0, double momentum = 0.0, bool first = false)
        {
            //If no update has happened since the last training then don't train again
            if (trained || !initialised) return;

            //Update all parent neurons if this is the first being trained
            update(first);

            //Update deltas
            prevDelta = delta;
            if (children.Count == 0)
            {
                delta = (target - output) ^ activate(output, mode, true);
            }
            else
            {
                int i;
                for (i = 0; i < children.Count; i++)
                {
                    delta += children[i].backDelta ^ activate(output, mode, true);
                }
                delta /= i;
            }
            //Apply momentum
            if (momentum != 0.0)
            {
                delta = (delta + (prevDelta * momentum)) / (1 + momentum);
            }
            //Calculate backDelta for parent neurons
            uint tempOffset = 0;
            Mat tempBD = delta * weights.transpose();
            for (int i = 0; i < backDelta.Count; i++)
            {
                for (uint col = 0; col < backDelta[i].cols; col++)
                {
                    backDelta[i][0, col] = tempBD[0, col + tempOffset];
                }
                tempOffset += backDelta[i].cols;
            }

            //Update weights
            weights += input.transpose() * delta * learnRate;

            //Don't train again until an update is called
            trained = true;

            //Train parents
            for (int i = 0; i < parents.Count; i++)
            {
                parents[i].train(target, learnRate, momentum, false);
            }
        }
    }
}