using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNet
{
    class Mat
    {
        protected uint _rows;
        protected uint _cols;
        protected double[,] store;

        public Mat(uint rows, uint cols, double init = 0)
        {
            _rows = rows;
            _cols = cols;
            store = new double[_rows, _cols];
            for (uint row = 0; row < _rows; row++)
            {
                for (uint col = 0; col < _cols; col++)
                {
                    store[row, col] = init;
                }
            }
        }

        public Mat(double[,] init)
        {
            _rows = (uint)init.GetLength(0);
            _cols = (uint)init.GetLength(1);
            store = new double[_rows, _cols];
            for (uint row = 0; row < _rows; row++)
            {
                for (uint col = 0; col < _cols; col++)
                {
                    store[row, col] = init[row, col];
                }
            }
        }

        public Mat(Mat cpy)
        {
            clone(cpy);
        }

        public void clone(Mat cpy)
        {
            _rows = cpy.rows;
            _cols = cpy.cols;
            store = cpy.store;
        }

        public uint rows
        {
            get { return _rows; }
        }

        public uint cols
        {
            get { return _cols; }
        }

        public double this[uint row, uint col]
        {
            get { return store[row, col]; }

            set { store[row, col] = value; }
        }

        /// <summary>
        /// Change the size of Mat A
        /// </summary>
        public void extend(uint rows, uint cols, double init = 0)
        {
            if (rows == _rows && cols == _cols) return;
            double[,] nStore = new double[rows, cols];
            for (uint row = 0; row < rows; row++)
            {
                for (uint col = 0; col < cols; col++)
                {
                    if (row < _rows && col < _cols)
                    {
                        nStore[row, col] = store[row, col];
                    }
                    else
                    {
                        nStore[row, col] = init;
                    }
                }
            }
            _rows = rows;
            _cols = cols;
            store = nStore;
        }

        /// <summary>
        /// Mat Transpose
        /// </summary>
        public Mat transpose()
        {
            Mat T = new Mat(_cols, _rows);
            for (uint row = 0; row < _rows; row++)
            {
                for (uint col = 0; col < _cols; col++)
                {
                    T[col, row] = store[row, col];
                }
            }
            return T;
        }

        /// <summary>
        /// Row Vector
        /// </summary>
        public Mat row(uint fromRow)
        {
            Mat T = new Mat(1, _cols);
            for (uint col = 0; col < _cols; col++)
            {
                T[0, col] = store[fromRow, col];
            }
            return T;
        }

        /// <summary>
        /// Column Vector
        /// </summary>
        public Mat column(uint fromCol)
        {
            Mat T = new Mat(_rows, 1);
            for (uint row = 0; row < _rows; row++)
            {
                T[row, 0] = store[row, fromCol];
            }
            return T;
        }

        /// <summary>
        /// Mat Transpose
        /// </summary>
        public Mat concat(Mat A, bool vert = false)
        {
            Mat T;
            if (!vert)
            {
                T = new Mat(_rows, _cols + A.cols);
                for (uint row = 0; row < T.rows; row++)
                {
                    for (uint col = 0; col < T.cols; col++)
                    {
                        T[row, col] = (col < _cols ? store[row, col] : A[row, col - _cols]);
                    }
                }
            }
            else
            {
                T = new Mat(_rows + A.rows, _cols);
                for (uint row = 0; row < T.rows; row++)
                {
                    for (uint col = 0; col < T.cols; col++)
                    {
                        T[row, col] = (row < _rows ? store[row, col] : A[row - _rows, col]);
                    }
                }
            }
            return T;
        }

        /// <summary>
        /// Mat Divide by Scalar
        /// </summary>
        public static Mat operator /(Mat A, double B)
        {
            return A * (1 / B);
        }

        /// <summary>
        /// Mat Multiplication by Scalar
        /// </summary>
        public static Mat operator *(Mat A, double B)
        {
            Mat T = new Mat(A.rows, A.cols);
            for (uint row = 0; row < T.rows; row++)
            {
                for (uint col = 0; col < T.cols; col++)
                {
                    T[row, col] = B * A[row, col];
                }
            }
            return T;
        }

        /// <summary>
        /// Mat Multiplication by Scalar
        /// </summary>
        public static Mat operator *(double A, Mat B)
        {
            return B * A;
        }

        /// <summary>
        /// Mat Multiplication by Scalar
        /// </summary>
        public static Mat operator *(Mat A, int B)
        {
            return A * (double)B;
        }

        /// <summary>
        /// Mat Multiplication by Scalar
        /// </summary>
        public static Mat operator *(int A, Mat B)
        {
            return B * (double)A;
        }

        /// <summary>
        /// Standard Mat Multiplication
        /// </summary>
        public static Mat operator *(Mat A, Mat B)
        {
            if (A.cols != B.rows) {
                return null; }

            Mat T = new Mat(A.rows, B.cols);
            for (uint row = 0; row < T.rows; row++)
            {
                for (uint col = 0; col < T.cols; col++)
                {
                    double i = 0.0;
                    for (uint pos = 0; pos < B.rows; pos++)
                    {
                        i += A[row, pos] * B[pos, col];
                    }
                    T[row, col] = i;
                }
            }
            return T;
        }

        /// <summary>
        /// Hadamard Mat Multiplication 
        /// </summary>
        public static Mat operator ^(Mat A, double B)
        {
            Mat T = new Mat(A.rows, A.cols, 1.0);
            for (uint row = 0; row < T.rows; row++)
            {
                for (uint col = 0; col < T.cols; col++)
                {
                    for (int i = 0; i < B; i++)
                    {
                        T[row, col] = A[row, col] * A[row, col];
                    }
                }
            }
            return T;
        }

        /// <summary>
        /// Hadamard Mat Multiplication 
        /// </summary>
        public static Mat operator ^(Mat A, Mat B)
        {
            if (A.rows != B.rows || A.cols != B.cols) { return null; }

            Mat T = new Mat(A.rows, A.cols);
            for (uint row = 0; row < T.rows; row++)
            {
                for (uint col = 0; col < T.cols; col++)
                {
                    T[row, col] = A[row, col] * B[row, col];
                }
            }
            return T;
        }

        /// <summary>
        /// Kronecker Mat Product 
        /// </summary>
        public static Mat operator %(Mat A, Mat B)
        {
            Mat T = new Mat(A.rows, B.cols);
            for (uint row = 0; row < T.rows; row++)
            {
                for (uint col = 0; col < T.cols; col++)
                {
                    //T[row, col] = A[row / B.rows, col / B.cols] * B[row % B.rows, col % B.cols];
                    T[row, col] = A[row, 0] * B[0, col];
                }
            }
            return T;
        }

        /// <summary>
        /// Mat Subtract
        /// </summary>
        public static Mat operator -(Mat A, Mat B)
        {
            if (A.rows != B.rows || A.cols != B.cols) { return null; }

            Mat T = new Mat(A.rows, A.cols);
            for (uint row = 0; row < T.rows; row++)
            {
                for (uint col = 0; col < T.cols; col++)
                {
                    T[row, col] = A[row, col] - B[row, col];
                }
            }
            return T;
        }

        /// <summary>
        /// Mat Addition
        /// </summary>
        public static Mat operator +(Mat A, Mat B)
        {
            if (A.rows != B.rows || A.cols != B.cols) { return null; }

            Mat T = new Mat(A.rows, A.cols);
            for (uint row = 0; row < T.rows; row++)
            {
                for (uint col = 0; col < T.cols; col++)
                {
                    T[row, col] = A[row, col] + B[row, col];
                }
            }
            return T;
        }

        /// <summary>
        /// Mat Component Sum
        /// </summary>
        public static double operator ~(Mat A)
        {
            double rtn = 0;
            for (uint row = 0; row < A.rows; row++)
            {
                for (uint col = 0; col < A.cols; col++)
                {
                    rtn += A[row, col];
                }
            }
            return rtn;
        }
    }

}
