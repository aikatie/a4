#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"

// Constants
static const double Delta = 0.001;
static const double xMid = 0.23701;
static const double yMid = 0.521;

// CUDA Kernel to compute the fractal
__global__ void computeFractal(unsigned char *d_pic, int width, int height, int num_frames, double Delta) {
    // Calculate thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    int frame = idx / total_pixels;  // Determine the frame
    int pixel_idx = idx % total_pixels;  // Determine the pixel within the frame

    if (frame < num_frames) {
        int row = pixel_idx / width;  // Calculate row
        int col = pixel_idx % width;  // Calculate column

        // Calculate fractal parameters
        double aspect_ratio = (double)width / height;
        double delta = Delta;
        for (int i = 0; i < frame; i++) {
            delta *= 0.98;  // Apply delta scaling for each frame
        }

        double x0 = xMid - delta * aspect_ratio;
        double y0 = yMid - delta;
        double dx = 2.0 * delta * aspect_ratio / width;
        double dy = 2.0 * delta / height;

        // Calculate the complex coordinate (cx, cy)
        double cx = x0 + col * dx;
        double cy = y0 + row * dy;

        double x = cx;
        double y = cy;
        int depth = 256;

        double x2, y2;
        // Compute the fractal
        do {
            x2 = x * x;
            y2 = y * y;
            y = 2 * x * y + cy;
            x = x2 - y2 + cx;
            depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));  // Escape condition

        // Store the depth value as color (mapping depth to 0-255)
        d_pic[frame * height * width + row * width + col] = (unsigned char)depth;
    }
}

int main(int argc, char *argv[]) {
    double start, end;

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

    // Allocate memory for image on host
    unsigned char *pic = (unsigned char *)malloc(num_frames * height * width * sizeof(unsigned char));
    if (pic == NULL) {
        fprintf(stderr, "Error: Unable to allocate memory on host.\n");
        exit(-1);
    }

    // Allocate memory for image on device
    unsigned char *d_pic;
    cudaMalloc((void **)&d_pic, num_frames * height * width * sizeof(unsigned char));

    // Configure CUDA kernel
    int th_per_blk = 256;  // Number of threads per block
    int num_blocks = (num_frames * height * width + th_per_blk - 1) / th_per_blk;  // Number of blocks

    // Start timing
    GET_TIME(start);

    // Launch the kernel
    computeFractal<<<num_blocks, th_per_blk>>>(d_pic, width, height, num_frames, Delta);

    // Synchronize to ensure the kernel finishes
    cudaDeviceSynchronize();

    // Stop timing
    GET_TIME(end);
    double elapsed = end - start;
    printf("CUDA compute time: %.4f s\n", elapsed);

    // Copy the result back to the host
    cudaMemcpy(pic, d_pic, num_frames * height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Write frames to BMP files (if size is manageable)
    if ((width <= 320) && (num_frames <= 100)) {
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic[frame * height * width], name);
        }
    }

    // Free memory
    free(pic);
    cudaFree(d_pic);

    return 0;
}
