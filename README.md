# Patch2Vec [[WNNA-2022 site]](http://rnns.net/wnna-2022/) [[Poster]](https://disk.yandex.ru/i/OK_mVsaQP_RemQ)
This is an official C# implementation for "Patch2Vec: a simple and efficient convolution algorithm for mobile neural networks" WNNA-2022 poster.

# Description
The meaning of most fast convolution algorithms, such as `im2col` or `im2row`, involves bringing the convolution to matrix multiplication, which allows optimizing memory access operations by using the processor cache. However, such methods either require a buffer for `srcC * kernelY * kernelH * dstH * dstW` elements, which is extremely irrational. The proposed `patch2vec` method unwraps each patch of the input image on the fly, and then applies all convolution filters to it. This implementation is not inferior in efficiency to classical algorithms like `im2col`, and in practice even surpasses them. The buffer for this algorithm will have the size of `srcC * kernelY * kernelX`, which is much smaller than in the case of similar methods. Moreover, `patch2vec` does not impose restrictions on the convolution parameters, unlike, for example, the Shmuel Vinograd method. The proposed algorithm is difficult to fit into classical machine learning frameworks due to the fact that they are focused on using GEMM as the core. Pure C#-based implementations make it easy to do this. For more detailed description, please see [Details.md](https://github.com/GlebSBrykin/Patch2Vec/blob/main/Details.md)

![Algorithm](https://github.com/GlebSBrykin/Patch2Vec/raw/main/Illustrations/conv2d.jpg)

# References
1. Lavin, Scott Gray: Fast Algorithms for Convolutional Neural Networks, arXiv preprint arXiv:1509.09308, 2015.
2. Anton V. Trusov, Elena E. Limonova, Dmitry P. Nikolaev and Vladimir V. Arlazarov: p-im2col: Simple Yet Efficient Convolution Algorithm With Flexibly Controlled Memory Overhead, IEEE Access PP(99):1-1 (2021)
