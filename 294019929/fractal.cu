// Alex Mena and Egor
// Alex forgot to write his name ;)

#include <stdlib.h>
#include <stdio.h>
#include <cuda.h>
#include "timer.h"
#include "fractal.h"

// Constants for the fractal
static const double Delta = 0.001;
static const double xMid = 0.23701;
static const double yMid = 0.521;

/**
 * CUDA kernel to compute the fractal.
 */
__global__ void computeFractalKernel(
    unsigned char *d_pic, int width, int height, int num_frames, double delta_base
) {
    
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_pixels = num_frames * width * height;

    if (pixel_idx >= total_pixels) return; // Boundary check

    int frame = pixel_idx / (width * height);
    int row = (pixel_idx % (width * height)) / width;
    int col = pixel_idx % width;

    double delta = delta_base * pow(0.98, frame);

    double aspect_ratio = (double)width / height;
    double x0 = xMid - delta * aspect_ratio;
    double y0 = yMid - delta;
    double dx = 2.0 * delta * aspect_ratio / width;
    double dy = 2.0 * delta / height;

    double cx = x0 + col * dx;
    double cy = y0 + row * dy;

    // Mandelbrot computation
    double x = cx, y = cy;
    int depth = 256;
    double x2, y2;

    do {
        x2 = x * x;
        y2 = y * y;
        y = 2 * x * y + cy;
        x = x2 - y2 + cx;
        depth--;
    } while ((depth > 0) && ((x2 + y2) < 5.0));

    d_pic[pixel_idx] = (unsigned char)depth;
}

int main(int argc, char *argv[]) {
    double start, end;

    printf("Fractal v1.6 [CUDA]\n");

    if (argc != 5) {
        fprintf(stderr, "usage: %s height width num_frames th_per_blk\n", argv[0]);
        exit(-1);
    }

    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int num_frames = atoi(argv[3]);
    int th_per_blk = atoi(argv[4]);

    if (width < 10 || height < 10 || num_frames < 1 || th_per_blk < 1) {
        fprintf(stderr, "Invalid parameters\n");
        exit(-1);
    }

    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

    size_t pic_size = num_frames * width * height * sizeof(unsigned char);

    // Host and device memory
    unsigned char *pic = (unsigned char *)malloc(pic_size);
    unsigned char *d_pic;
    cudaMalloc(&d_pic, pic_size);

    GET_TIME(start);

    // Total number of pixels
    int total_pixels = num_frames * width * height;

    // Number of blocks and threads
    int num_blocks = (total_pixels + th_per_blk - 1) / th_per_blk;

    // Launch kernel
    computeFractalKernel<<<num_blocks, th_per_blk>>>(d_pic, width, height, num_frames, Delta);

    // Copy results back to host
    cudaMemcpy(pic, d_pic, pic_size, cudaMemcpyDeviceToHost);

    GET_TIME(end);

    printf("CUDA compute time: %.4f s\n", end - start);

    // Write BMP files if conditions are met
    if ((width <= 320) && (num_frames <= 100)) {
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic[frame * width * height], name);
        }
    }

    // Cleanup
    free(pic);
    cudaFree(d_pic);

    return 0;
}