
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#include <iostream>
#define TILE_WIDTH 24

namespace mxnet
{
namespace op
{


__global__ void forward_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.
    We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    */

    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]
    #define x4d(i3,i2,i1,i0) x[(i3) * (C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]
    #define k4d(i3,i2,i1,i0) k[(i3) * (C * K * K) + (i2)*(K * K) + (i1)*(K) + i0]

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;
    

    int W_grid = (W_out-1/TILE_WIDTH)+1;
    //int H_grid = (H_out-1/TILE_WIDTH)+1;

    int b = blockIdx.x;
    int m = blockIdx.y;

    int h = blockIdx.z / W_grid + threadIdx.y;
    int w = blockIdx.z % W_grid + threadIdx.x;
    int p, q;
    float acc = 0.0;

    for (p = 0; p < K; p++)
         for (q = 0; q < K; q++)
             acc +=  x4d(b, 0, (h + p), (w + q)) * k4d(m, 0, p, q);

    y4d(b, m, h, w) = acc;

    #undef y4d
    #undef x4d
    #undef k4d
    
}

__global__ void unroll_kernel(float *y, const float *x, const float *k, float *X_unroll, const int B, const int M, const int C, const int H, const int W, const int K) {

    #define x4d(i3,i2,i1,i0) x[(i3) * (B * H * W) + (i2)*(H * W) + (i1)*(W) + i0]

    int b, s, h_out, w_out, h_unroll, p, q;
    
    int t = blockIdx.x * TILE_WIDTH * TILE_WIDTH + threadIdx.x;        // 24*24 threads in total
    int H_out = H-K+1;
    int W_out = W-K + 1;
    int W_unroll = H_out * W_out;

    if (t < B * W_unroll) {
        b = t / W_unroll;                                              // which color, maximum number 10k
        s = t % W_unroll;                                              // which column in the color, maximum number:24*24
        h_out = s / W_out;                                             // in vertical direction which 5*5 it belongs 
        w_out = s % W_out;                                             // in horizontal direction which 5*5 it belongs
        //h_unroll = h_out * W_out + w_out;                              // matrix of the size 24*24*25*100000
        for(p = 0; p < K; p++){
            for(q = 0; q < K; q++) {
                h_unroll = p * K + q;           
                X_unroll[h_unroll*B * W_unroll+ t] = x4d(b, 0, h_out + p, w_out + q);
            }
        }
    }
	#undef x4d
}

__global__ void matrixMultiplyShared(const float *A, float *X_unroll, float *Result, const int B, const int M, const int C, const int H, const int W, const int K) {
  __shared__ float subTileA[TILE_WIDTH][TILE_WIDTH];
  __shared__ float subTileB[TILE_WIDTH][TILE_WIDTH];
  int bx = blockIdx.x;  int by = blockIdx.y;
  int tx = threadIdx.x;  int ty = threadIdx.y;
  int numAColumns = K*K;
  int numBColumns = B*TILE_WIDTH*TILE_WIDTH;
  int numARows = M;
  int numBRows = K*K;
  int numCColumns = numBColumns;
  int numCRows = numARows;
  
  int Row = by* TILE_WIDTH + ty;
  int Col = bx* TILE_WIDTH + tx;
  float Pvalue = 0;
  
  for (int m = 0; m < (numAColumns-1)/TILE_WIDTH+1; m++){
    if(Row < numARows && m*TILE_WIDTH+tx < numAColumns){
      subTileA[ty][tx] = A[Row*numAColumns + m*TILE_WIDTH +tx];
    }else{
      subTileA[ty][tx]=0;
    }
    if(Col < numBColumns && m*TILE_WIDTH+ty < numBRows){
      subTileB[ty][tx] = X_unroll[(m*TILE_WIDTH+ty)*numBColumns+Col];
    }else{
      subTileB[ty][tx] = 0;
    }
    __syncthreads();
    if(Row < numCRows && Col < numCColumns){
      for (int k = 0; k< TILE_WIDTH; k++)
        Pvalue +=subTileA[ty][k] * subTileB[k][tx];
    }
    __syncthreads();
    
  }
  if(Row < numCRows && Col < numCColumns){
    Result[Row*numCColumns+Col] = Pvalue;
  } 
}

__global__ void mapping(float *y, float *Result, const int B, const int M, const int C, const int H, const int W, const int K) {

    #define y4d(i3,i2,i1,i0) y[(i3) * (M * H_out * W_out) + (i2)*(H_out * W_out) + (i1)*(W_out) + i0]

    int b, s, dy, dx, m;
    
    int t = blockIdx.x * TILE_WIDTH * TILE_WIDTH + threadIdx.x;        // 24*24 threads in total
    int H_out = TILE_WIDTH;
    int W_out = TILE_WIDTH;
    int W_unroll = H_out * W_out;

    if (t < B * W_unroll) {
        b = t / W_unroll;                                              // which color, maximum number 10k
        s = t % W_unroll;                                              // which column in the color, maximum number:24*24
		dy = s/TILE_WIDTH;
		dx = s % TILE_WIDTH;
        for(m = 0; m < M; m++){          
			y4d(b,m,dy,dx) = Result[m*B*TILE_WIDTH*TILE_WIDTH+(b*TILE_WIDTH*TILE_WIDTH+s)];
        }
    }
	#undef y4d
}


// This function is called by new-inl.h
// Any code you write should be executed by this function
//For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
template<>
void forward<gpu,float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    // cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    // ...

    const int B = x.shape_[0]; 
    const int M = y.shape_[1]; 
    const int H = x.shape_[2]; 
    const int W = x.shape_[3]; 
    const int K = w.shape_[3];

    float* X_unroll;
	float* Result;
    int H_out = H-K+1;
    int W_out = W-K+1;
	int num_blocks = 10000;
    cudaMalloc((void **)&X_unroll, (H_out*W_out*K*K*B)*sizeof(float));
	cudaMalloc((void **)&Result, (H_out*W_out*M*B)*sizeof(float));

    // Set the kernel dimensions
    dim3 gridDim(B, M, ceil((W - K + 1)/TILE_WIDTH) * ceil((H - K + 1)/TILE_WIDTH));
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
	
	dim3 gridDim2(B*TILE_WIDTH,ceil(M/TILE_WIDTH),1); 
    dim3 blockDim2(TILE_WIDTH, TILE_WIDTH, 1);
	
    // Call the kernel
	unroll_kernel<<<num_blocks, TILE_WIDTH>>>(y.dptr_,x.dptr_,w.dptr_,X_unroll, B,M,1,H,W,K);
    matrixMultiplyShared<<<gridDim2,blockDim2>>>(w.dptr_, X_unroll, Result, B,M,1,H,W,K);	
	mapping<<<num_blocks, TILE_WIDTH * TILE_WIDTH>>>(y.dptr_,Result,B,M,1,H,W,K);
	
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}


/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif