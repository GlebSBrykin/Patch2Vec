# Patch2Vec algorithm details
**Here we suppose that all convolutions are two-dimensional.**

## Naive algorithm

Consider the layout of tensors in memory using the example of a matrix.

![Matrix layout](https://github.com/GlebSBrykin/Patch2Vec/raw/main/Illustrations/tensor%20layout.jpg)

Multidimensional arrays are stored in memory sequentially as a continuous bytes block. It is important to note that matrices are stored line by line, three-dimensional tensors (arrays of matrices) are stored matrix by matrix, etc.

Given the naive (i.e. as in conventional convolution definition) implementation of conv2d operation. The convolution operation is a sliding window of the convolution kernel, at each position of which, for each kernel from the set, the sum of the element products of the convolution kernel values and the corresponding values of the input image area is calculated. This operation is clearly shown in the animation below.

![Conv2d animation](https://github.com/GlebSBrykin/Patch2Vec/raw/main/Illustrations/conv2d%20animation.gif)

Let's look at this process deeper.

![Naive conv2d](https://github.com/GlebSBrykin/Patch2Vec/raw/main/Illustrations/naive%20conv2d.jpg)

When performing memory operations, the processor loads a certain amount of data behind the requested address into the cache. Read-ahead allows the processor to perform operations on data and load/unload them from/to RAM in parallel, instead of making a request to RAM at each iteration. Note that the processor cache is faster memory than RAM, so you should distribute memory access in such a way as to minimize RAM accesses by maximizing cache accesses. The illustration of the naive convolution above shows memory accesses and cache misses that occur due to a request for memory addresses, data from which has not been loaded into the cache. On modern computing devices, the performance of the naive algorithm in cases when number of kernels is larger than 1 is primarily limited by the bandwidth of RAM, which is relatively small even for the most modern types.

The source code of naive conv2d used in experiments is shown below:

```C#
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
    if(group > dstC)
    {
        for(int b = 0; b < batch; ++b)
        {
            var src1 = src + b * group * srcC * srcH * srcW;
            var dst1 = dst + b * group * dstC * dstH * dstW;
            Parallel.For(0, group, (int g) =>
            {
                var src2 = src1 + g * srcC * srcH * srcW;
                var weight1 = weight + g * dstC * srcC * kernelY * kernelX;
                var dst2 = dst1 + g * dstC * dstH * dstW;
                var bias1 = bias + g * dstC;
                for(int dc = 0; dc < dstC; ++dc)
                {
                    var weight2 = weight1 + dc * srcC * kernelY * kernelX;
                    var dst3 = dst2 + dc * dstH * dstW;
                    for(int dy = 0; dy < dstH; ++dy)
                    {
                        var dst4 = dst3 + dy * dstW;
                        for(int dx = 0; dx < dstW; ++dx)
                        {
                            float sum = 0;
                            for(int sc = 0; sc < srcC; ++sc)
                            {
                                var src3 = src2 + sc * srcH * srcW;
                                var weight3 = weight2 + sc * kernelY * kernelX;
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
                                        if((sx >= 0) && (sx < srcW))
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
            var src1 = src + b * group * srcC * srcH * srcW;
            var dst1 = dst + b * group * dstC * dstH * dstW;
            for(int g = 0; g < group; ++g)
            {
                var src2 = src1 + g * srcC * srcH * srcW;
                var weight1 = weight + g * dstC * srcC * kernelY * kernelX;
                var dst2 = dst1 + g * dstC * dstH * dstW;
                var bias1 = bias + g * dstC;
                Parallel.For(0, dstC, (int dc) =>
                {
                    var weight2 = weight1 + dc * srcC * kernelY * kernelX;
                    var dst3 = dst2 + dc * dstH * dstW;
                    for(int dy = 0; dy < dstH; ++dy)
                    {
                        var dst4 = dst3 + dy * dstW;
                        for(int dx = 0; dx < dstW; ++dx)
                        {
                            float sum = 0;
                            for(int sc = 0; sc < srcC; ++sc)
                            {
                                var src3 = src2 + sc * srcH * srcW;
                                var weight3 = weight2 + sc * kernelY * kernelX;
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
                                        if((sx >= 0) && (sx < srcW))
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
```

## Im2Col

Im2Col is conventional practical-usable algorithm for all cases of convolution.

![Im2Col](https://github.com/GlebSBrykin/Patch2Vec/raw/main/Illustrations/im2col.jpg)

The idea of im2col is to group the values of the input image required to perform the convolution operation, which allows you to perform a non-optimal operation of loading an image patch once, and not D times (where D is the number of convolution cores). The convolution operation is reduced to matrix multiplication, which can be easily optimized using the processor cache and many other approaches, such as SIMD.

The source code of im2col conv2d used in experiments is shown below:

```C#
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
        im2col(src, srcC, srcH, srcW, kernelY, kernelX, dilationY, dilationX, strideY, strideX, padY, padX, padH, padW, buf);
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
```

## Patch2Vec

Patch2Vec is a modified im2col algorithm. The main difference between patch2vec and im2col is that patch2vec does not store all image patches in memory, instead, all convolution kernels are applied to each patch extracted into a continuous vector, after which the obtained values are written to the corresponding position of the output image. Patch2vec allows to perform a non-optimal operation of extracting the values of the image area only once, after which all convolution kernels are applied to the vector stored in memory sequentially, using the processor cache as much as possible. The performance of patch2vec is comparable to that of im2col, however, patch2vec requires `srcC * kernelY * kernelX * sizeof(dtype)` bytes of memory for the buffer per thread, and im2col requires `dstH * dstW * srcC * kernelY * kernelX * sizeof(dtype)` bytes.

![Patch2Vec](https://github.com/GlebSBrykin/Patch2Vec/raw/main/Illustrations/patch2vec.jpg)

The source code of patch2vec conv2d used in experiments is shown below:

```C#
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
    for(int b = 0; b < batch; ++b)
    {
        var src1 = src + b * group * srcC * srcH * srcW;
        var dst1 = dst + b * group * dstC * dstH * dstW;
        for(int g = 0; g < group; ++g)
        {
            var src2 = src1 + g * srcC * srcH * srcW;
            var dst2 = dst1 + g * dstC * dstH * dstW;
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
                        var src3 = src2 + sc * srcH * srcW;
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
                        dst4[dc * dstH * dstW] = sum + bias[g * dstC + dc];
                    }
                }
            });
        }
    }
}
```

## Sources
* Conv2d animation: https://ai-news.ru/2018/07/kak_rabotaet_svertochnaya_nejronnaya_set_arhitektura_primery_osobennosti.html
* Im2Col illustration: https://zybuluo.com/Team/note/1175439
