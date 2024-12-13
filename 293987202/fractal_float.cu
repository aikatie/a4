//To calculate speedup, uncomment the serial implementation and print statements for printing the serial time and speedup

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "timer.h"
#include "fractal.h"

static const float Delta = 0.001;
static const float xMid = 0.23701;
static const float yMid = 0.521;

__global__ void fractalKernel(unsigned char *pic, int width, int height, int num_frames, float delta, float aspect_ratio) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int frame = blockIdx.z; 

    __shared__ unsigned char sharedDepth[128][128];

     if (row < height && col < width) {
        float current_delta = delta * pow(0.98, frame);
        float x0 = xMid - current_delta * aspect_ratio;
        float y0 = yMid - current_delta;
        float dx = 2.0 * current_delta * aspect_ratio / width;
        float dy = 2.0 * current_delta / height;

        float cx = x0 + col * dx;
        float cy = y0 + row * dy;
        float x = cx;
        float y = cy;
        int depth = 256;

        float x2, y2;
        do {
            x2 = x * x;
            y2 = y * y;
            y = 2 * x * y + cy;
            x = x2 - y2 + cx;
            depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));
        sharedDepth[threadIdx.y][threadIdx.x] = (unsigned char)depth;
        __syncthreads();
        int idx = frame * height * width + row * width + col;
        pic[idx] = sharedDepth[threadIdx.y][threadIdx.x];
    }
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s height width num_frames\n", argv[0]);
        exit(-1);
    }

    if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
    int width = atoi(argv[1]);
    if (width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
    int height = atoi(argv[2]);
    if (height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
    int num_frames = atoi(argv[3]);
    if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

    unsigned char *pic_serial = (unsigned char *)malloc(num_frames * height * width * sizeof(unsigned char));
    unsigned char *pic_cuda = (unsigned char *)malloc(num_frames * height * width * sizeof(unsigned char));

    // serial imp
    double start_serial, end_serial;
    GET_TIME(start_serial);

    float delta = Delta;
    const float aspect_ratio = (float)width / height;

 /*   for (int frame = 0; frame < num_frames; frame++) {
        const float x0 = xMid - delta * aspect_ratio;
        const float y0 = yMid - delta;
        const float dx = 2.0 * delta * aspect_ratio / width;
        const float dy = 2.0 * delta / height;

        for (int row = 0; row < height; row++) {
            const float cy = y0 + row * dy;
            for (int col = 0; col < width; col++) {
                const float cx = x0 + col * dx;
                float x = cx;
                float y = cy;
                int depth = 256;
                float x2, y2;
                do {
                    x2 = x * x;
                    y2 = y * y;
                    y = 2 * x * y + cy;
                    x = x2 - y2 + cx;
                    depth--;
                } while ((depth > 0) && ((x2 + y2) < 5.0));

                pic_serial[frame * height * width + row * width + col] = (unsigned char)depth;
            }
        }
        delta *= 0.98;
    }
*/
    GET_TIME(end_serial);
    double elapsed_serial = end_serial - start_serial;
//    printf("Serial compute time: %.4f s\n", elapsed_serial);

    // cuda imp
    double start_cuda, end_cuda;
    unsigned char *d_pic;
    cudaMalloc(&d_pic, num_frames * height * width * sizeof(unsigned char));
    cudaError_t err = cudaMalloc(&d_pic, num_frames * height * width * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return -1;
    }
    GET_TIME(start_cuda);

    dim3 threadsPerBlock(128, 128);
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   num_frames);

    fractalKernel<<<numBlocks, threadsPerBlock>>>(d_pic, width, height, num_frames, Delta, aspect_ratio);
    cudaDeviceSynchronize();
    cudaMemcpy(pic_cuda, d_pic, num_frames * height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    GET_TIME(end_cuda);
    double elapsed_cuda = end_cuda - start_cuda;
    printf("Parallel compute time: %.6f s\n", elapsed_cuda);

    if ((width <= 10000) && (num_frames <= 100)) {
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic_cuda[frame * height * width], name);
        }
    }
//    printf("Speedup: %.4f\n",elapsed_serial/elapsed_cuda);
    free(pic_serial);
    free(pic_cuda);
    cudaFree(d_pic);

    return 0;
}
