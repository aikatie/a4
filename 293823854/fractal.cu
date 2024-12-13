#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "fractal.h"
#include "timer.h"

#define BLOCK_SIZE 16 // Use 16x16 threads per block for 2D grid

// CUDA Kernel for fractal computation
__global__ void computeFractal(unsigned char *pic, int width, int height, double x0, double y0, double dx, double dy, int frame) {
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Get row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Get column index

    if (row < height && col < width) {
        int idx = row * width + col;

        double cx = x0 + col * dx;
        double cy = y0 + row * dy;

        double x = cx;
        double y = cy;
        int depth = 256;

        double x2, y2;
        do {
            x2 = x * x;
            y2 = y * y;
            y = 2 * x * y + cy;
            x = x2 - y2 + cx;
            depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));

        pic[frame * width * height + idx] = (unsigned char)depth;
    }
}

int main(int argc, char *argv[]) {
    printf("Fractal v1.6 [CUDA - Optimized]\n");

    // Add device properties check here
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Using CUDA Device: %s with %d multiprocessors\n", prop.name, prop.multiProcessorCount);

    if (argc != 4) {
        fprintf(stderr, "Usage: %s height width num_frames\n", argv[0]);
        exit(-1);
    }

    int height = atoi(argv[1]);
    int width = atoi(argv[2]);
    int numFrames = atoi(argv[3]);

    if (width < 10 || height < 10 || numFrames < 1) {
        fprintf(stderr, "Error: width and height must be at least 10, and numFrames at least 1\n");
        exit(-1);
    }

    printf("Computing %d frames of %d by %d fractal\n", numFrames, width, height);

    // Allocate pinned memory for the host
    unsigned char *h_pic;
    cudaHostAlloc((void **)&h_pic, numFrames * height * width * sizeof(unsigned char), cudaHostAllocDefault);

    // Allocate memory on the device
    unsigned char *d_pic;
    cudaMalloc(&d_pic, numFrames * height * width * sizeof(unsigned char));

    // Constants
    const double xMid = 0.23701;
    const double yMid = 0.521;
    const double Delta = 0.001;

    // CUDA Event timers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CUDA Stream for overlapping computation and memory transfers
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEventRecord(start); // Start the timer

    // Compute frames
    for (int frame = 0; frame < numFrames; frame++) {
        double delta = Delta * pow(0.98, frame); // Compute delta for this frame

        // Precompute constants on the host
        double aspect_ratio = (double)width / height;
        double x0 = xMid - delta * aspect_ratio;
        double y0 = yMid - delta;
        double dx = 2.0 * delta * aspect_ratio / width;
        double dy = 2.0 * delta / height;

        // Configure 2D grid and block dimensions
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

        // Launch kernel
        computeFractal<<<numBlocks, threadsPerBlock, 0, stream>>>(d_pic, width, height, x0, y0, dx, dy, frame);

        // Copy result back to host asynchronously
        cudaMemcpyAsync(&h_pic[frame * height * width], &d_pic[frame * height * width],
                        height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost, stream);
    }

    // Ensure all computations and transfers are complete
    cudaStreamSynchronize(stream);

    cudaEventRecord(stop); // Stop the timer
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Total computation took %f milliseconds\n", milliseconds);

    // Write BMP files for verification
    if ((width <= 320) && (numFrames <= 100)) { // Avoid large files
        for (int frame = 0; frame < numFrames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &h_pic[frame * height * width], name);
        }
    }

    // Free memory
    cudaFreeHost(h_pic);
    cudaFree(d_pic);

    cudaStreamDestroy(stream);
    printf("Fractal computation completed.\n");
    return 0;
}
