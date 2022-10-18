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

## Im2Col

Im2Col is conventional practical-usable algorithm for all cases of convolution.

![Im2Col](https://github.com/GlebSBrykin/Patch2Vec/raw/main/Illustrations/im2col.jpg)

The idea of im2col is to group the values of the input image required to perform the convolution operation, which allows you to perform a non-optimal operation of loading an image patch once, and not D times (where D is the number of convolution cores). The convolution operation is reduced to matrix multiplication, which can be easily optimized using the processor cache and many other approaches, such as SIMD.

## Patch2Vec

Patch2Vec is a modified im2col algorithm. The main difference between patch2vec and im2col is that patch2vec does not store all image patches in memory, instead, all convolution kernels are applied to each patch extracted into a continuous vector, after which the obtained values are written to the corresponding position of the output image. Patch2vec allows to perform a non-optimal operation of extracting the values of the image area only once, after which all convolution kernels are applied to the vector stored in memory sequentially, using the processor cache as much as possible. The performance of patch2vec is comparable to that of im2col, however, patch2vec requires `srcC * kernelY * kernelX * sizeof(dtype)` bytes of memory for the buffer per thread, and im2col requires `dstH * dstW * srcC * kernelY * kernelX * sizeof(dtype)` bytes.

![Patch2Vec](https://github.com/GlebSBrykin/Patch2Vec/raw/main/Illustrations/patch2vec.jpg)
