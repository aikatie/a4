#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"

static const double Delta = 0.001;
static const double xMid = 0.23701;
static const double yMid = 0.521;

// CUDA kernel for fractal computation
__global__ void computeFractalKernel(unsigned char* pic, int width, int height, 
                                   double x0, double y0, double dx, double dy,
                                   int frame, int frame_offset) {
    // Calculate global thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        const double cx = x0 + col * dx;
        const double cy = y0 + row * dy;
        
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
        
        pic[frame_offset + row * width + col] = (unsigned char)depth;
    }
}

int main(int argc, char *argv[]) {
    printf("Fractal v1.6 [CUDA]\n");

    // Read command line arguments
    if (argc != 4) {
        fprintf(stderr, "usage: %s height width num_frames\n", argv[0]);
        exit(-1);
    }
    int width = atoi(argv[1]);
    if (width < 10) {
        fprintf(stderr, "error: width must be at least 10\n");
        exit(-1);
    }
    int height = atoi(argv[2]);
    if (height < 10) {
        fprintf(stderr, "error: height must be at least 10\n");
        exit(-1);
    }
    int num_frames = atoi(argv[3]);
    if (num_frames < 1) {
        fprintf(stderr, "error: num_frames must be at least 1\n");
        exit(-1);
    }
    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

    // Allocate host memory
    unsigned char *h_pic = (unsigned char*)malloc(num_frames * height * width * sizeof(unsigned char));

    // Allocate device memory
    unsigned char *d_pic;
    cudaMalloc(&d_pic, num_frames * height * width * sizeof(unsigned char));

    // Define block and grid dimensions
    dim3 threadsPerBlock(16, 16);  // 256 threads per block
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Start timing
    double start, end;
    GET_TIME(start);

    // Compute frames
    const double aspect_ratio = (double)width / height;
    double delta = Delta;

    for (int frame = 0; frame < num_frames; frame++) {
        // Calculate frame-specific parameters
        const double x0 = xMid - delta * aspect_ratio;
        const double y0 = yMid - delta;
        const double dx = 2.0 * delta * aspect_ratio / width;
        const double dy = 2.0 * delta / height;
        
        // Launch kernel
        int frame_offset = frame * height * width;
        computeFractalKernel<<<numBlocks, threadsPerBlock>>>(
            d_pic, width, height, x0, y0, dx, dy, frame, frame_offset
        );
        
        // Update zoom factor for next frame
        delta *= 0.98;
    }

    // Synchronize and check for errors
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Copy result back to host
    cudaMemcpy(h_pic, d_pic, num_frames * height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // End timing
    GET_TIME(end);
    double elapsed = end - start;
    printf("CUDA compute time: %.6f s\n", elapsed);

    // Write frames to BMP files
    if ((width <= 320) && (num_frames <= 100)) {
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &h_pic[frame * height * width], name);
        }
    }

    // Cleanup
    cudaFree(d_pic);
    free(h_pic);

    return 0;
}