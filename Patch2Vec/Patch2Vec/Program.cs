//***************************************************************************************************
//* (C) Gleb S. Brykin, 2022.
//***************************************************************************************************

using System;
using System.Threading.Tasks;
using System.Runtime.InteropServices;

namespace Patch2Vec
{

    public unsafe static class Program
    {

        /// <summary>
        /// Basic implementation of two-dimensional convolution.
        /// </summary>
        /// <param name="src">Source data.</param>
        /// <param name="batch">Batch size.</param>
        /// <param name="srcC">Input channels.</param>
        /// <param name="srcH">Input height.</param>
        /// <param name="srcW">Input width.</param>
        /// <param name="kernelY">Kernel height.</param>
        /// <param name="kernelX">Kernel width.</param>
        /// <param name="dilationY">Dilation of the kernel by height.</param>
        /// <param name="dilationX">Dilation of the kernel by width.</param>
        /// <param name="strideY">Stride of the convolution by height.</param>
        /// <param name="strideX">Stride of the convolution by width.</param>
        /// <param name="padY">Zero padding by left side.</param>
        /// <param name="padX">Zero padding by top side.</param>
        /// <param name="padH">Zero padding by right side.</param>
        /// <param name="padW">Zero padding by bottom side.</param>
        /// <param name="group">Convolution groups. If group=srcC=dstC, convolution is depthwise separable.</param>
        /// <param name="weight">Weights (kernels).</param>
        /// <param name="bias">Bias.</param>
        /// <param name="dst">Destination memory.</param>
        /// <param name="dstC">Output channels.</param>
        public static void NaiveConv2d(float* src,
                                       int batch,
                                       int srcC,
                                       int srcH,
                                       int srcW,
                                       int kernelY,
                                       int kernelX,
                                       int dilationY,
                                       int dilationX,
                                       int strideY,
                                       int strideX,
                                       int padY,
                                       int padX,
                                       int padH,
                                       int padW,
                                       int group,
                                       float* weight,
                                       float* bias,
                                       float* dst,
                                       int dstC)
        {
            int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            dstC = dstC / group;
            srcC = srcC / group;
            var srcH_srcW = srcH * srcW;
            var dstH_dstW = dstH * dstW;
            var srcC_srcH_srcW = srcC * srcH_srcW;
            var dstC_dstH_dstW = dstC * dstH_dstW;
            var group_srcC_srcH_srcW = group * srcC_srcH_srcW;
            var group_dstC_dstH_dstW = group * dstC_dstH_dstW;
            var kernelY_kernelX = kernelY * kernelX;
            var srcC_kernelY_kernelX = srcC * kernelY_kernelX;
            var dstC_srcC_kernelY_kernelX = dstC * srcC_kernelY_kernelX;
            if(group > dstC)
            {
                for(int b = 0; b < batch; ++b)
                {
                    var src1 = src + b * group_srcC_srcH_srcW;
                    var dst1 = dst + b * group_dstC_dstH_dstW;
                    Parallel.For(0, group, (int g) =>
                    {
                        var src2 = src1 + g * srcC_srcH_srcW;
                        var weight1 = weight + g * dstC_srcC_kernelY_kernelX;
                        var dst2 = dst1 + g * dstC_dstH_dstW;
                        var bias1 = bias + g * dstC;
                        for(int dc = 0; dc < dstC; ++dc)
                        {
                            var weight2 = weight1 + dc * srcC_kernelY_kernelX;
                            var dst3 = dst2 + dc * dstH_dstW;
                            for(int dy = 0; dy < dstH; ++dy)
                            {
                                var dst4 = dst3 + dy * dstW;
                                for(int dx = 0; dx < dstW; ++dx)
                                {
                                    float sum = 0;
                                    for(int sc = 0; sc < srcC; ++sc)
                                    {
                                        var src3 = src2 + sc * srcH_srcW;
                                        var weight3 = weight2 + sc * kernelY_kernelX;
                                        for(int ky = 0; ky < kernelY; ++ky)
                                        {
                                            var weight4 = weight3 + ky * kernelX;
                                            int sy = dy * strideY + ky * dilationY - padY;
                                            if((sy < 0) || (sy >= srcH))
                                            {
                                                continue;
                                            }
                                            var src4 = src3 + sy * srcW;
                                            for(int kx = 0; kx < kernelX; ++kx)
                                            {
                                                int sx = dx * strideX + kx * dilationX - padX;
                                                if(sx >= 0 && sx < srcW)
                                                {
                                                    sum += src4[sx] * weight4[kx];
                                                }
                                            }
                                        }
                                    }
                                    dst4[dx] = sum + bias1[dc];
                                }
                            }
                        }
                    });
                }
            }
            else
            {
                for(int b = 0; b < batch; ++b)
                {
                    var src1 = src + b * group_srcC_srcH_srcW;
                    var dst1 = dst + b * group_dstC_dstH_dstW;
                    for(int g = 0; g < group; ++g)
                    {
                        var src2 = src1 + g * srcC_srcH_srcW;
                        var weight1 = weight + g * dstC_srcC_kernelY_kernelX;
                        var dst2 = dst1 + g * dstC_dstH_dstW;
                        var bias1 = bias + g * dstC;
                        Parallel.For(0, dstC, (int dc) =>
                        {
                            var weight2 = weight1 + dc * srcC_kernelY_kernelX;
                            var dst3 = dst2 + dc * dstH_dstW;
                            for(int dy = 0; dy < dstH; ++dy)
                            {
                                var dst4 = dst3 + dy * dstW;
                                for(int dx = 0; dx < dstW; ++dx)
                                {
                                    float sum = 0;
                                    for(int sc = 0; sc < srcC; ++sc)
                                    {
                                        var src3 = src2 + sc * srcH_srcW;
                                        var weight3 = weight2 + sc * kernelY_kernelX;
                                        for(int ky = 0; ky < kernelY; ++ky)
                                        {
                                            var weight4 = weight3 + ky * kernelX;
                                            int sy = dy * strideY + ky * dilationY - padY;
                                            if((sy < 0) || (sy >= srcH))
                                            {
                                                continue;
                                            }
                                            var src4 = src3 + sy * srcW;
                                            for(int kx = 0; kx < kernelX; ++kx)
                                            {
                                                int sx = dx * strideX + kx * dilationX - padX;
                                                if(sx >= 0 && sx < srcW)
                                                {
                                                    sum += src4[sx] * weight4[kx];
                                                }
                                            }
                                        }
                                    }
                                    dst4[dx] = sum + bias1[dc];
                                }
                            }
                        });
                    }
                }
            }
        }

        /// <summary>
        /// Image to column conversion.
        /// </summary>
        /// <param name="src">Source data.</param>
        /// <param name="srcC">Input channels.</param>
        /// <param name="srcH">Input height.</param>
        /// <param name="srcW">Input width.</param>
        /// <param name="kernelY">Kernel height.</param>
        /// <param name="kernelX">Kernel width.</param>
        /// <param name="dilationY">Dilation of the kernel by height.</param>
        /// <param name="dilationX">Dilation of the kernel by width.</param>
        /// <param name="strideY">Stride of the convolution by height.</param>
        /// <param name="strideX">Stride of the convolution by width.</param>
        /// <param name="padY">Zero padding by left side.</param>
        /// <param name="padX">Zero padding by top side.</param>
        /// <param name="padH">Zero padding by right side.</param>
        /// <param name="padW">Zero padding by bottom side.</param>
        /// <param name="buf">Buffer.</param>
        public static void im2col(float* src,
                                  int srcC,
                                  int srcH,
                                  int srcW,
                                  int kernelY,
                                  int kernelX,
                                  int dilationY,
                                  int dilationX, 
                                  int strideY,
                                  int strideX,
                                  int padY,
                                  int padX,
                                  int padH,
                                  int padW,
                                  float* buf)
        {
            int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            for(int sc = 0; sc < srcC; ++sc)
            {
                var scsrcH = sc * srcH;
                for(int ky = 0; ky < kernelY; ++ky)
                {
                    int sy_ = ky * dilationY - padY;
                    for(int kx = 0; kx < kernelX; ++kx)
                    {
                        int sx_ = kx * dilationX - padX;
                        for(int dy = 0; dy < dstH; ++dy)
                        {
                            int sy = sy_ + dy * strideY;
                            if((sy < 0) || (sy >= srcH))
                            {
                                for(int dx = 0; dx < dstW; ++dx)
                                {
                                    *buf++ = 0;
                                }
                                continue;
                            }
                            var src1 = src + (scsrcH + sy) * srcW;
                            for(int dx = 0; dx < dstW; ++dx)
                            {
                                int sx = sx_ + dx * strideX;
                                if((sx >= 0) && (sx < srcW))
                                {
                                    *buf++ = src1[sx];
                                }
                                else
                                {
                                    *buf++ = 0;
                                }
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Matrix multiplication.
        /// </summary>
        /// <param name="M">A rows.</param>
        /// <param name="N">A columns.</param>
        /// <param name="K">B columns.</param>
        /// <param name="A">Left matrix.</param>
        /// <param name="B">Right matrix.</param>
        /// <param name="C">Result matrix.</param>
        public static void mm(int M,
                              int N,
                              int K,
                              float* A,
                              float* B,
                              float* C)
        {
            Parallel.For(0, M, (int i) =>
            {
                var Cp = C + i * N;
                var Ap = A + i * K;
                for(int j = 0; j < N; ++j)
                {
                    Cp[j] = 0;
                }
                for(int k = 0; k < K; ++k)
                {
                    var a = Ap[k];
                    var Bp = B + k * N;
                    for(int j = 0; j < N; ++j)
                    {
                        Cp[j] += a * Bp[j];
                    }
                }
            });
        }

        /// <summary>
        /// Im2Col-based implementation of two-dimensional convolution.
        /// </summary>
        /// <param name="src">Source data.</param>
        /// <param name="batch">Batch size.</param>
        /// <param name="srcC">Input channels.</param>
        /// <param name="srcH">Input height.</param>
        /// <param name="srcW">Input width.</param>
        /// <param name="kernelY">Kernel height.</param>
        /// <param name="kernelX">Kernel width.</param>
        /// <param name="dilationY">Dilation of the kernel by height.</param>
        /// <param name="dilationX">Dilation of the kernel by width.</param>
        /// <param name="strideY">Stride of the convolution by height.</param>
        /// <param name="strideX">Stride of the convolution by width.</param>
        /// <param name="padY">Zero padding by left side.</param>
        /// <param name="padX">Zero padding by top side.</param>
        /// <param name="padH">Zero padding by right side.</param>
        /// <param name="padW">Zero padding by bottom side.</param>
        /// <param name="group">Convolution groups. If group=srcC=dstC, convolution is depthwise separable.</param>
        /// <param name="weight">Weights (kernels).</param>
        /// <param name="bias">Bias.</param>
        /// <param name="dst">Destination memory.</param>
        /// <param name="dstC">Output channels.</param>
        public static void Im2ColConv2d(float* src,
                                        int batch,
                                        int srcC,
                                        int srcH,
                                        int srcW,
                                        int kernelY,
                                        int kernelX,
                                        int dilationY,
                                        int dilationX,
                                        int strideY,
                                        int strideX,
                                        int padY,
                                        int padX,
                                        int padH,
                                        int padW,
                                        int group,
                                        float* weight,
                                        float* bias,
                                        float* dst,
                                        int dstC)
        {
            int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            int M = dstC / group;
            int N = dstH * dstW;
            int K = srcC * kernelY * kernelX / group;
            var buf = (float*)Marshal.AllocCoTaskMem(srcC * kernelY * kernelX * dstH * dstW * sizeof(float));
            for(int b = 0; b < batch; ++b)
            {
                im2col(src, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX,
                       strideY, strideX, padY, padX, padH, padW, buf);
                for(int g = 0; g < group; ++g)
                {
                    mm(M, N, K, weight + M * K * g, buf + N * K * g, dst + M * N * g);
                }
                for(int i = 0; i < dstC; ++i)
                {
                    var pdst = dst + i * N;
                    for(int j = 0; j < N; ++j)
                    {
                        pdst[j] += bias[i];
                    }
                }
                src += srcC * srcH * srcW;
                dst += dstC * dstH * dstW;
            }
            Marshal.FreeCoTaskMem((IntPtr)buf);
        }

        /// <summary>
        /// Patch2Vec-based implementation of two-dimensional convolution.
        /// </summary>
        /// <param name="src">Source data.</param>
        /// <param name="batch">Batch size.</param>
        /// <param name="srcC">Input channels.</param>
        /// <param name="srcH">Input height.</param>
        /// <param name="srcW">Input width.</param>
        /// <param name="kernelY">Kernel height.</param>
        /// <param name="kernelX">Kernel width.</param>
        /// <param name="dilationY">Dilation of the kernel by height.</param>
        /// <param name="dilationX">Dilation of the kernel by width.</param>
        /// <param name="strideY">Stride of the convolution by height.</param>
        /// <param name="strideX">Stride of the convolution by width.</param>
        /// <param name="padY">Zero padding by left side.</param>
        /// <param name="padX">Zero padding by top side.</param>
        /// <param name="padH">Zero padding by right side.</param>
        /// <param name="padW">Zero padding by bottom side.</param>
        /// <param name="group">Convolution groups. If group=srcC=dstC, convolution is depthwise separable.</param>
        /// <param name="weight">Weights (kernels).</param>
        /// <param name="bias">Bias.</param>
        /// <param name="dst">Destination memory.</param>
        /// <param name="dstC">Output channels.</param>
        public static void Patch2VecConv2d(float* src,
                                           int batch,
                                           int srcC,
                                           int srcH,
                                           int srcW,
                                           int kernelY,
                                           int kernelX,
                                           int dilationY,
                                           int dilationX,
                                           int strideY,
                                           int strideX,
                                           int padY,
                                           int padX,
                                           int padH,
                                           int padW,
                                           int group,
                                           float* weight,
                                           float* bias,
                                           float* dst,
                                           int dstC)
        {
            int dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            int dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            dstC = dstC / group;
            srcC = srcC / group;
            var srcCkernelYkernelX = srcC * kernelY * kernelX;
            var srcH_srcW = srcH * srcW;
            var dstH_dstW = dstH * dstW;
            var srcC_srcH_srcW = srcC * srcH_srcW;
            var dstC_dstH_dstW = dstC * dstH_dstW;
            var group_srcC_srcH_srcW = group * srcC_srcH_srcW;
            var group_dstC_dstH_dstW = group * dstC_dstH_dstW;
            for(int b = 0; b < batch; ++b)
            {
                var src1 = src + b * group_srcC_srcH_srcW;
                var dst1 = dst + b * group_dstC_dstH_dstW;
                for(int g = 0; g < group; ++g)
                {
                    var src2 = src1 + g * srcC_srcH_srcW;
                    var dst2 = dst1 + g * dstC_dstH_dstW;
                    Parallel.For(0, dstH, (int dy) =>
                    {
                        var dst3 = dst2 + dy * dstW;
                        int sy_ = dy * strideY - padY;
                        var buffer = stackalloc float[srcCkernelYkernelX];
                        for(int dx = 0; dx < dstW; ++dx)
                        {
                            var dst4 = dst3 + dx;
                            int sx_ = dx * strideX - padX;
                            for(int sc = 0; sc < srcC; ++sc)
                            {
                                var src3 = src2 + sc * srcH_srcW;
                                for(int ky = 0; ky < kernelY; ++ky)
                                {
                                    int sy = sy_ + ky * dilationY;
                                    if((sy < 0) || (sy >= srcH))
                                    {
                                        for(int kx = 0; kx < kernelX; ++kx)
                                        {
                                            *buffer++ = 0;
                                        }
                                        continue;
                                    }
                                    var src4 = src3 + sy * srcW;
                                    for(int kx = 0; kx < kernelX; ++kx)
                                    {
                                        int sx = sx_ + kx * dilationX;
                                        if((sx >= 0) && (sx < srcW))
                                        {
                                            *buffer++ = src4[sx];
                                        }
                                        else
                                        {
                                            *buffer++ = 0;
                                        }
                                    }
                                }
                            }
                            buffer -= srcCkernelYkernelX;
                            for(int dc = 0; dc < dstC; ++dc)
                            {
                                float sum = 0;
                                var pweight = weight + (g * dstC + dc) * srcCkernelYkernelX;
                                for(int i = 0; i < srcCkernelYkernelX; ++i)
                                {
                                    sum += buffer[i] * pweight[i];
                                }
                                dst4[dc * dstH_dstW] = sum + bias[g * dstC + dc];
                            }
                        }
                    });
                }
            }
        }

        /// <summary>
        /// Random number generator.
        /// </summary>
        public static Random random = new Random();

        /// <summary>
        /// Creates the floating point array of size "size" and fills it with random values in [0..1] range.
        /// </summary>
        /// <param name="size">Size.</param>
        /// <returns>Pointer to array.</returns>
        public static float* rand(int size)
        {
            var mem = (float*)Marshal.AllocCoTaskMem(size * sizeof(float));
            for(int i = 0; i < size; ++i)
            {
                mem[i] = (float)random.NextDouble();
            }
            return mem;
        }

        /// <summary>
        /// Mean Absolete Error between two arrays.
        /// </summary>
        /// <param name="a">First array.</param>
        /// <param name="b">Second array.</param>
        /// <param name="size">Size (should be same for a and b).</param>
        /// <returns>MAE.</returns>
        public static float mae(float* a, float* b, int size)
        {
            float diff = 0;
            for(int i = 0; i < size; ++i)
            {
                diff += Math.Abs(a[i] - b[i]);
            }
            return diff / size;
        }

        public static void Main()
        {
            // Hyperparameters
            const int batch = 1;
            const int srcC = 64;
            const int srcH = 128;
            const int srcW = 128;
            const int kernelY = 5;
            const int kernelX = 5;
            const int dilationY = 1;
            const int dilationX = 1;
            const int strideY = 1;
            const int strideX = 1;
            const int padY = 1;
            const int padX = 1;
            const int padH = 1;
            const int padW = 1;
            const int group = 1;
            const int dstC = 128;
            // Output sizes
            var dstH = (srcH + padY + padH - (dilationY * (kernelY - 1) + 1)) / strideY + 1;
            var dstW = (srcW + padX + padW - (dilationX * (kernelX - 1) + 1)) / strideX + 1;
            // Input data, weights, biases
            var src = rand(batch * srcC * srcH * srcW);
            var weight = rand((dstC / group) * (srcC / group) * kernelY * kernelX);
            var bias = rand(dstC);
            // Destination storages for naive, im2col and patch2vec convolutions
            var dst1 = (float*)Marshal.AllocCoTaskMem(batch * dstC * dstH * dstW * sizeof(float));
            var dst2 = (float*)Marshal.AllocCoTaskMem(batch * dstC * dstH * dstW * sizeof(float));
            var dst3 = (float*)Marshal.AllocCoTaskMem(batch * dstC * dstH * dstW * sizeof(float));
            // Naive
            var ms = DateTime.Now;
            NaiveConv2d(src, batch, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group, weight, bias, dst1, dstC);
            Console.WriteLine(string.Format("Naive: {0} ms", (DateTime.Now - ms).TotalMilliseconds));
            // Im2Col
            ms = DateTime.Now;
            Im2ColConv2d(src, batch, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group, weight, bias, dst2, dstC);
            Console.WriteLine(string.Format("Im2Col: {0} ms", (DateTime.Now - ms).TotalMilliseconds));
            // Check error between Naive and Im2Col results. Should be 0.
            Console.WriteLine(string.Format("MAE(Naive, Im2Col): {0}", mae(dst1, dst2, batch * dstC * dstH * dstW)));
            // Patch2Vec
            ms = DateTime.Now;
            Patch2VecConv2d(src, batch, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, group, weight, bias, dst3, dstC);
            Console.WriteLine(string.Format("Patch2Vec: {0} ms", (DateTime.Now - ms).TotalMilliseconds));
            // Check error between Naive and Patch2Vec results. Should be 0.
            Console.WriteLine(string.Format("MAE(Naive, Patch2Vec): {0}", mae(dst1, dst3, batch * dstC * dstH * dstW)));
            Console.Write("Press any key to continue . . . ");
            Console.ReadKey(true);
        }
    }
}