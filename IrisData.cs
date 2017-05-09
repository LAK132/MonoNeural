using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;

namespace NeuralNet
{
    class IrisData
    {
        public Network.dataSet irisTest;

        public static Mat[] stringToMat(string input, int[] inputWidths)
        {
            List<List<double>>[] inputs = new List<List<double>>[inputWidths.Length];
            for (int i = 0; i < inputs.Length; i++)
                inputs[i] = new List<List<double>>();
            int width = 0;
            for (int i = 0; i < inputWidths.Length; i++)
                width += inputWidths[i];
            Regex rgx = new Regex(@"(\S+)");
            MatchCollection matches = rgx.Matches(input);
            if (matches.Count > 0)
            {
                int pos = 0;
                foreach (Match match in matches)
                {
                    int row = pos / width;
                    int col = pos - (width * row);
                    int sel = 0;
                    for (int testPos = 0; inputWidths[sel] + testPos <= col;)
                    {
                        testPos += inputWidths[sel];
                        sel++;
                    }

                    while (inputs[sel].Count <= row)
                        inputs[sel].Add(new List<double>());
                    inputs[sel][row].Add(double.Parse(match.Value));
                    pos++;
                }
            }

            Mat[] A = new Mat[inputs.Length];
            for (int i = 0; i < inputs.Length; i++)
            {
                A[i] = new Mat((uint)inputs[i].Count, (uint)inputs[i][0].Count);
                for (uint row = 0; row < inputs[i].Count; row++)
                {
                    for (uint col = 0; col < inputs[i][0].Count; col++)
                    {
                        A[i][row, col] = inputs[i][(int)row][(int)col];
                    }
                }
            }
            return A;
        }

        public void stringIn(string input, int inputWidth, int outputWidth, ref Mat inputMat, ref Mat outputMat)
        {
            Mat[] A = stringToMat(input, new int[2]{ inputWidth, outputWidth });
            inputMat = A[0];
            outputMat = A[1];
        }

        public IrisData()
        {
            string binary = @"
0 0 0 1 1 0 0
0 0 0 0 0 0 0
0 0 1 1 1 0 0 
0 0 1 0 0 0 0
0 1 0 1 1 0 0
0 1 0 0 0 0 0
1 0 0 1 1 0 0
1 0 0 0 0 0 0
";

            string test = @"
0.420705	0.269205	0.50625	0.463986	0	1	0
0.720948	0.480944	0.763306	0.828996	0	0	1
0.928809	0.763262	0.944758	0.865497	0	0	1
0.628566	0.410364	0.778427	0.828996	0	0	1
0.697853	0.480944	0.717944	0.901998	0	0	1
0.305227	0.586813	0.143347	0.135476	1	0	0
0.697853	0.516233	0.596976	0.573489	0	1	0
0.259036	0.480944	0.143347	0.171977	1	0	0
0.328323	0.657392	0.143347	0.171977	1	0	0
0.259036	0.516233	0.173589	0.135476	1	0	0
0.328323	0.763262	0.218952	0.208478	1	0	0
0.466896	0.975	0.158468	0.208478	1	0	0
0.259036	0.622103	0.218952	0.135476	1	0	0
0.466896	0.763262	0.18871	0.171977	1	0	0
0.443801	0.445654	0.476008	0.536988	0	1	0
0.60547	0.586813	0.642339	0.646491	0	1	0
0.859522	0.410364	0.854032	0.755994	0	0	1
0.60547	0.375074	0.672581	0.719493	0	0	1
0.282131	0.304495	0.612097	0.682992	0	0	1
0.60547	0.622103	0.778427	0.938499	0	0	1
0.397609	0.727972	0.158468	0.135476	1	0	0
0.466896	0.304495	0.687702	0.792495	0	0	1
0.443801	0.480944	0.612097	0.60999	0	1	0
0.212844	0.622103	0.143347	0.171977	1	0	0
0.651661	0.480944	0.763306	0.719493	0	0	1
0.328323	0.304495	0.385282	0.463986	0	1	0
0.60547	0.445654	0.778427	0.719493	0	0	1
0.697853	0.586813	0.793548	0.828996	0	0	1
0.212844	0.692682	0.0828629	0.135476	1	0	0
0.628566	0.551523	0.733065	0.901998	0	0	1
0.559279	0.480944	0.627218	0.573489	0	1	0
0.536183	0.622103	0.612097	0.646491	0	1	0
0.928809	0.339785	0.975	0.901998	0	0	1
0.328323	0.586813	0.18871	0.24498	1	0	0
0.536183	0.445654	0.612097	0.60999	0	1	0
0.466896	0.410364	0.551613	0.536988	0	1	0
0.651661	0.480944	0.808669	0.865497	0	0	1
0.790235	0.480944	0.82379	0.828996	0	0	1

";

            int inputWidth = 4;
            int outputWidth = 3;

            stringIn(test, inputWidth, outputWidth, ref irisTest.inputs, ref irisTest.outputs);

            irisTest.classes = Network.outputToClasses(irisTest.outputs);
            irisTest.count = irisTest.inputs.rows;

            /*
            public struct dataSet
            {
                public Matrix inputs;
                public Matrix outputs;
                public Matrix classes;
                public uint count;
                public Matrix bias;
            }*/
        }
    }
}
