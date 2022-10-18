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

When performing memory operations, the processor loads a certain amount of data behind the requested address into the cache. Read-ahead allows the processor to perform operations on data and load/unload them from/to RAM in parallel, instead of making a request to RAM at each iteration. Note that the processor cache is faster memory than RAM, so you should distribute memory access in such a way as to minimize RAM accesses by maximizing cache accesses. The illustration of the naive convolution above shows memory accesses and cache misses that occur due to a request for memory addresses, data from which has not been loaded into the cache. On modern computing devices, the performance of the naive algorithm is primarily limited by the bandwidth of RAM, which is relatively small even for the most modern types.

## Im2Col

Im2Col is conventional practical-usable algorithm for all cases of convolution.

![Im2Col](https://github.com/GlebSBrykin/Patch2Vec/raw/main/Illustrations/im2col.jpg)

