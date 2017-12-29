
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>
#define TILE_WIDTH 24
#define TILE_WIDTH_M 25
#define TILE_WIDTH_W 32
#define K_SIZE 5
#define K_SIZE_SQ 25
#define NUM_K 50
#define numCColumns 5760000
#define out_dim 576
#define in_dim 784

__constant__ float cM[K_SIZE_SQ*NUM_K];

namespace mxnet
{
namespace op
{


__global__ void unroll_kernel(const float *x, const float *k, float *X_unroll, const int B, const int W) {

    #define x4d(i3,i2,i1,i0) x[(i3) * (in_dim) + (i2)*(in_dim) + (i1)*(W) + i0]

    int b, s, h_out, w_out, h_unroll, p, q;
    
    int t = blockIdx.x * out_dim + threadIdx.x;        // 24*24 threads in a block
    //int H_out = H-K+1;
    //int W_out = W-K + 1;
    //int W_unroll = H_out * W_out;

    //if (t < B * W_unroll) {
        b = t / out_dim;                                              // which color, maximum number 10k
        s = t % out_dim;                                              // which column in the color, maximum number:24*24
        h_out = s / TILE_WIDTH;                                             // in vertical direction which 5*5 it belongs 
        w_out = s % TILE_WIDTH;                                             // in horizontal direction which 5*5 it belongs
        //h_unroll = h_out * W_out + w_out; 
                                     // matrix of the size 24*24*25*100000
        int base = B * out_dim;
        for(p = 0; p < K_SIZE; p++){
            for(q = 0; q < K_SIZE; q++) {
                h_unroll = p * K_SIZE + q;           
                X_unroll[h_unroll*base+ t] = x4d(b, 0, h_out + p, w_out + q);
            }
        }
    //}
    #undef x4d
}


__global__ void matrixMultiplyShared(float *X_unroll, float *Result) {
    //__shared__ float subTileA[TILE_WIDTH_M][TILE_WIDTH_W];
    __shared__ float subTileB[TILE_WIDTH_M][TILE_WIDTH_W];
    #define y4d(i3,i2,i1,i0) Result[(i3) * (NUM_K * out_dim) + (i2)*(out_dim) + (i1)*(TILE_WIDTH) + i0]
    
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x;  int ty = threadIdx.y;


    //int numAColumns = K*K;
    //int numBColumns = B*out_dim;
    //int numARows = M;
    //int numBRows = K*K;
    //int numCColumns = numBColumns;
    //int numCRows = M;

    int Row = by* TILE_WIDTH_M + ty;
    int Col = bx* TILE_WIDTH_W + tx;
    float Pvalue = 0;

    int b = Col/out_dim;
    int m = Row;
    int sub_idx = Col%out_dim;
    int dy = sub_idx/TILE_WIDTH;
    int dx = sub_idx%TILE_WIDTH;
        
    //if(Col < numBColumns && ty < numBRows){
    subTileB[ty][tx] = X_unroll[ty*numCColumns +Col];
    //}else{
    //    subTileB[ty][tx] = 0;
    //}
    __syncthreads();
    
    int base = Row*K_SIZE_SQ;
    //if(Row < numCRows && Col < numCColumns){
    for (int k = 0; k< TILE_WIDTH_M; k++){
         Pvalue+=cM[base + k]*subTileB[k][tx];
    }
    y4d(b,m,dy,dx) = Pvalue;
    //}
    //__syncthreads();
    

    //if(Row < numCRows && Col < numCColumns){
    //Result[Row*numCColumns+Col] = Pvalue;
    //}

    //int linear_idx = Row*numCColumns+Col;
    
    #undef y4d
}

/*
__global__ void mapping(float *y, float *Result, const int B, const int M) {

    #define y4d(i3,i2,i1,i0) y[(i3) * (M * out_dim) + (i2)*(out_dim) + (i1)*(TILE_WIDTH) + i0]

    int b, s, dy, dx, m;
    
    int t = blockIdx.x * out_dim + threadIdx.x;        // 24*24 threads in total
    //int H_out = TILE_WIDTH;
    //int W_out = TILE_WIDTH;
    //int W_unroll = H_out * W_out;

    //if (t < B * W_unroll) {
        b = t / out_dim;                                              // which color, maximum number 10k
        s = t % out_dim;                                              // which column in the color, maximum number:24*24
        dy = s/TILE_WIDTH;
        dx = s %TILE_WIDTH;
        int base = b*out_dim+s;
        for(m = 0; m < M; m++){          
            y4d(b,m,dy,dx) = Result[m*numCColumns+base];
        }
    //}
    #undef y4d
}
*/

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    // cudaStream_t s = y.stream_->stream_;

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0]; 
    const int M = y.shape_[1]; 
    const int H = x.shape_[2]; 
    const int W = x.shape_[3]; 
    const int K = w.shape_[3]; 

    // Set the kernel dimensions
    //int x_tile_width = TILE_WIDTH + K - 1;
    
    // call unroll
    float* X_unroll;
    //float* Result;
    //int H_out = H-K+1;
    //int W_out = W-K+1;
    //int out_dim = TILE_WIDTH*TILE_WIDTH;

    cudaMalloc((void **)&X_unroll, (out_dim*K_SIZE_SQ*B)*sizeof(float));
    //cudaMalloc((void **)&Result, (numCColumns*M)*sizeof(float));

    dim3 gridDim_unroll(B,1,1); 
    dim3 blockDim_unroll(TILE_WIDTH*TILE_WIDTH, 1, 1);
    unroll_kernel<<<gridDim_unroll, blockDim_unroll>>>(x.dptr_,w.dptr_,X_unroll, B,W);


    // Call matrixmul kernel
    cudaMemcpyToSymbol(cM, w.dptr_, K_SIZE_SQ*NUM_K * sizeof(float));
    dim3 gridDim((numCColumns-1)/TILE_WIDTH_W+1, (M-1)/TILE_WIDTH_M+1, 1); 
    dim3 blockDim(TILE_WIDTH_W, TILE_WIDTH_M, 1);
    matrixMultiplyShared<<<gridDim,blockDim>>>(X_unroll, y.dptr_); 

    //mapping<<<gridDim_unroll, blockDim_unroll>>>(y.dptr_,Result,B,M);

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