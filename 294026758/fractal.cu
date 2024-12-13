#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fractal.h"
#include "timer.h"

static const double Delta = 0.001;
static const double xMid = 0.23701;
static const double yMid = 0.521;

// CUDA kernel for fractal computation
__global__ void computeFractal(unsigned char *pic, int width, int height, double *deltas, double xMid, double yMid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx >= total_pixels) return;

    int frame = blockIdx.y; // Each frame is handled by a separate block in the y dimension
    int pixel_idx = idx;
    int row = pixel_idx / width;
    int col = pixel_idx % width;

    // Get delta for the current frame
    double delta = deltas[frame];

    // Calculate aspect ratio
    double aspect_ratio = (double)width / height;

    // Calculate coordinates
    double x0 = xMid - delta * aspect_ratio;
    double y0 = yMid - delta;
    double dx = 2.0 * delta * aspect_ratio / width;
    double dy = 2.0 * delta / height;

    double cx = x0 + col * dx;
    double cy = y0 + row * dy;

    // Fractal calculation
    double x = cx, y = cy;
    double x2, y2;
    int depth = 256;

    do {
        x2 = x * x;
        y2 = y * y;
        y = 2 * x * y + cy;
        x = x2 - y2 + cx;
        depth--;
    } while ((depth > 0) && ((x2 + y2) < 5.0));

    pic[frame * total_pixels + pixel_idx] = (unsigned char)depth;
}

int main(int argc, char *argv[]) {
    double start, end;

    printf("Fractal v1.6 [CUDA]\n");

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

    // Total number of pixels
    int total_pixels = width * height;

    // Allocate memory for the image array
    unsigned char *pic = (unsigned char *)malloc(num_frames * total_pixels * sizeof(unsigned char));
    unsigned char *d_pic;

    // Allocate device memory
    cudaMalloc(&d_pic, num_frames * total_pixels * sizeof(unsigned char));

    // Precompute deltas
    double *deltas = (double *)malloc(num_frames * sizeof(double));
    double *d_deltas;
    cudaMalloc(&d_deltas, num_frames * sizeof(double));

    for (int frame = 0; frame < num_frames; frame++) {
        deltas[frame] = Delta * pow(0.98, frame);
    }
    cudaMemcpy(d_deltas, deltas, num_frames * sizeof(double), cudaMemcpyHostToDevice);

    // Start timer
    GET_TIME(start);

    // Define thread and block sizes
    int th_per_blk = 256;
    int num_blocks = (total_pixels + th_per_blk - 1) / th_per_blk;
    dim3 grid(num_blocks, num_frames); // Separate grid dimension for frames

    // Launch the kernel
    computeFractal<<<grid, th_per_blk>>>(d_pic, width, height, d_deltas, xMid, yMid);

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy the result back to host memory
    cudaMemcpy(pic, d_pic, num_frames * total_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // End timer
    GET_TIME(end);
    double elapsed = end - start;
    printf("CUDA compute time: %.6f s\n", elapsed);

    // Write frames to BMP files
    if ((width <= 320) && (num_frames <= 100)) {
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic[frame * total_pixels], name);
        }
    }

    // Free memory
    free(pic);
    free(deltas);
    cudaFree(d_pic);
    cudaFree(d_deltas);

    return 0;
}

