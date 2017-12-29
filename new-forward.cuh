
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <iostream>
#define TILE_WIDTH 24.0

namespace mxnet
{
namespace op
{


template<typename gpu, typename DType>
__global__ void forward_kernel(DType *y, const DType *x, const DType *k, const int B, const int M, const int C, const int H, const int W, const int K, int W_grid,int H_grid, const int H_out, const int W_out) {

    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

     //int b, m, h, w, c, p, q;
     int b, m, h, w,  p, q;
     b = blockIdx.x;
     m = blockIdx.y;
     h = blockIdx.z / W_grid + threadIdx.y;
     w = blockIdx.z % W_grid + threadIdx.x;
     float acc = 0.0;
//     for (c = 0; c < C; c++) { // sum over all input channels
         for (p = 0; p < K; p++) // loop over KxK filter
             for (q = 0; q < K; q++)
                 //acc +=  x4d(b, c, (h + p), (w + q)) * k4d(m, c, p, q);
                 acc +=  x4d(b, 0, (h + p), (w + q)) * k4d(m, 0, p, q);
//     }
     y4d(b, m, h, w) = acc;
    
    #undef y4d
    #undef x4d
    #undef k4d
}




// This function is called by new-inl.h
// Any code you write should be executed by this function
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &k) {
    

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
     //cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    const int B = x.shape_[0]; // B - batches
    const int M = y.shape_[1]; // Number of output feature map
    //const int C = x.shape_[1]; // Number of input feature map
    const int H = x.shape_[2]; // Height of each input map image
    const int W = x.shape_[3]; // width of each input map image
    const int K = k.shape_[3]; // Height of each filter bank. Assumption is filters are type square. 

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    
    int W_grid = ceil(W_out/TILE_WIDTH);
    int H_grid = ceil(H_out/TILE_WIDTH);
    //int W_grid = (W_out/TILE_WIDTH);
    //int H_grid = (H_out/TILE_WIDTH);
    int Z = H_grid * W_grid;

    // Set the kernel dimensions
    dim3 gridDim(B, M, Z); // Batches, output feature map, Z ...)
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

    // Call the kernel
    //forward_kernel<gpu, DType><<<gridDim, blockDim, 0, s>>>(y.dptr_,x.dptr_,k.dptr_, B,M,C,H,W,K, W_grid, H_grid, H_out, W_out);
    //forward_kernel<gpu, DType><<<gridDim, blockDim>>>(y.dptr_,x.dptr_,k.dptr_, B,M,C,H,W,K, W_grid, H_grid, H_out, W_out);
    forward_kernel<gpu, DType><<<gridDim, blockDim>>>(y.dptr_,x.dptr_,k.dptr_, B,M,1,H,W,K, W_grid, H_grid, H_out, W_out);

    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}



}
}

#endif