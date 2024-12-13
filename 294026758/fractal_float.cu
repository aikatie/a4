#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fractal.h"
#include "timer.h"

static const float Delta = 0.001f;
static const float xMid = 0.23701f;
static const float yMid = 0.521f;

// CUDA kernel for fractal computation
__global__ void computeFractal(unsigned char *pic, int width, int height, float *deltas, float xMid, float yMid) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;

    if (idx >= total_pixels) return;

    int frame = blockIdx.y; // Each frame is handled by a separate block in the y dimension
    int pixel_idx = idx;
    int row = pixel_idx / width;
    int col = pixel_idx % width;

    // Get delta for the current frame
    float delta = deltas[frame];

    // Calculate aspect ratio
    float aspect_ratio = (float)width / height;

    // Calculate coordinates
    float x0 = xMid - delta * aspect_ratio;
    float y0 = yMid - delta;
    float dx = 2.0f * delta * aspect_ratio / width;
    float dy = 2.0f * delta / height;

    float cx = x0 + col * dx;
    float cy = y0 + row * dy;

    // Fractal calculation
    float x = cx, y = cy;
    float x2, y2;
    int depth = 256;

    do {
        x2 = x * x;
        y2 = y * y;
        y = 2 * x * y + cy;
        x = x2 - y2 + cx;
        depth--;
    } while ((depth > 0) && ((x2 + y2) < 5.0f));

    pic[frame * total_pixels + pixel_idx] = (unsigned char)depth;
}

int main(int argc, char *argv[]) {
    printf("Fractal v1.6 [CUDA] - Float Version\n");

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
    cudaError_t err = cudaMalloc(&d_pic, num_frames * total_pixels * sizeof(unsigned char));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error for d_pic: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Precompute deltas
    float *deltas = (float *)malloc(num_frames * sizeof(float));
    float *d_deltas;
    err = cudaMalloc(&d_deltas, num_frames * sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc error for d_deltas: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    for (int frame = 0; frame < num_frames; frame++) {
        deltas[frame] = Delta * powf(0.98f, frame);
    }
    cudaMemcpy(d_deltas, deltas, num_frames * sizeof(float), cudaMemcpyHostToDevice);

    // CUDA Event timing
    cudaEvent_t startEvent, stopEvent;
    float elapsedTime;

    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    // Start the event
    cudaEventRecord(startEvent);

    // Define thread and block sizes
    int th_per_blk = 256;
    int num_blocks = (total_pixels + th_per_blk - 1) / th_per_blk;
    dim3 grid(num_blocks, num_frames); // Separate grid dimension for frames

    // Launch the kernel
    computeFractal<<<grid, th_per_blk>>>(d_pic, width, height, d_deltas, xMid, yMid);

    // Check for kernel errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA kernel launch error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // Synchronize to make sure the kernel finishes before proceeding
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA synchronization error: %s\n", cudaGetErrorString(err));
        exit(-1);
    }

    // End the event
    cudaEventRecord(stopEvent);
    cudaEventSynchronize(stopEvent);

    // Calculate elapsed time
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);
    printf("CUDA compute time: %.6f s\n", elapsedTime / 1000.0f); // Convert to seconds

    // Copy the result back to host memory
    cudaMemcpy(pic, d_pic, num_frames * total_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

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

    // Destroy CUDA events
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    return 0;
}

