#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include "fractal.h"
#include "timer.h"

static const float Delta = 0.001f;  // Changed to float
static const float xMid = 0.23701f; // Changed to float
static const float yMid = 0.521f;   // Changed to float

// CUDA Kernel to compute the fractal
__global__ void fractalKernel(unsigned char *pic, int width, int height, int num_frames, float delta_init) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Pixel column
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Pixel row
    int frame = blockIdx.z;                          // Frame index

    if (col >= width || row >= height || frame >= num_frames) return;

    // Compute delta for the current frame
    float delta = delta_init * powf(0.98f, frame); // Changed pow to powf for float precision

    const float aspect_ratio = (float)width / height;
    const float x0 = xMid - delta * aspect_ratio;
    const float y0 = yMid - delta;
    const float dx = 2.0f * delta * aspect_ratio / width;
    const float dy = 2.0f * delta / height;

    const float cx = x0 + col * dx;
    const float cy = y0 + row * dy;

    float x = cx;
    float y = cy;
    int depth = 256;

    float x2, y2;
    do {
        x2 = x * x;
        y2 = y * y;
        y = 2.0f * x * y + cy;
        x = x2 - y2 + cx;
        depth--;
    } while ((depth > 0) && ((x2 + y2) < 5.0f)); // Changed 5.0 to 5.0f

    // Store the result
    pic[frame * height * width + row * width + col] = (unsigned char)depth;
}

int main(int argc, char *argv[]) {
    float start, end;

    printf("Fractal v1.6 [float CUDA]\n");

    // Read command-line arguments
    if (argc != 4) {
        fprintf(stderr, "usage: %s width height num_frames\n", argv[0]);
        exit(-1);
    }
    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int num_frames = atoi(argv[3]);

    if (width < 10 || height < 10 || num_frames < 1) {
        fprintf(stderr, "Error: Invalid dimensions or number of frames.\n");
        exit(-1);
    }

    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

    size_t total_pixels = num_frames * width * height;

    // Allocate memory for the image on host and device
    unsigned char *pic_host = (unsigned char *)malloc(total_pixels * sizeof(unsigned char));
    unsigned char *pic_device;
    cudaMalloc((void **)&pic_device, total_pixels * sizeof(unsigned char));

    // Start timing
    GET_TIME(start);

    // Configure CUDA kernel
    dim3 th_per_blk(16, 16);
    dim3 blocksPerGrid((width + th_per_blk.x - 1) / th_per_blk.x,
                       (height + th_per_blk.y - 1) / th_per_blk.y,
                       num_frames);

    fractalKernel<<<blocksPerGrid, th_per_blk>>>(pic_device, width, height, num_frames, Delta);
    cudaDeviceSynchronize();

    // Copy the results back to the host
    cudaMemcpy(pic_host, pic_device, total_pixels * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Stop timing
    GET_TIME(end);
    float elapsed = end - start;
    printf("Float CUDA compute time: %.4f s\n", elapsed);

    // Write BMP files (if small enough)
    if (width <= 320 && num_frames <= 100) {
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "float_fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic_host[frame * height * width], name);
        }
    }

    free(pic_host);
    cudaFree(pic_device);

    return 0;
}
