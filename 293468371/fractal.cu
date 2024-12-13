#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include "fractal.h"
#include "timer.h"

static const double Delta = 0.001;
static const double xMid = 0.23701;
static const double yMid = 0.521;

__global__ void computeFractal(unsigned char *pic, int width, int height, int num_frames, double delta) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height * num_frames;

    if (idx < total_pixels) {
        int frame = idx / (width * height);
        int pixel_idx = idx % (width * height);
        int row = pixel_idx / width;
        int col = pixel_idx % width;

        const double aspect_ratio = (double)width / height;
        const double x0 = xMid - delta * aspect_ratio;
        const double y0 = yMid - delta;
        const double dx = 2.0 * delta * aspect_ratio / width;
        const double dy = 2.0 * delta / height;

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

        pic[frame * height * width + row * width + col] = (unsigned char)depth;
    }
}

int main(int argc, char *argv[]) {
    double start, end;

    printf("Fractal v1.7 [cuda]\n");

    /* read command line arguments */
    if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
    int width = atoi(argv[1]);
    if (width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
    int height = atoi(argv[2]);
    if (height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
    int num_frames = atoi(argv[3]);
    if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

    unsigned char *h_pic = (unsigned char*)malloc(num_frames * height * width * sizeof(unsigned char));
    unsigned char *d_pic;
    cudaMalloc((void**)&d_pic, num_frames * height * width * sizeof(unsigned char));

    /* start time */
    GET_TIME(start);

 /* Compute frames */
    double delta = Delta;
    int blockSize = 256; // Threads per block
    int numBlocks = (width * height + blockSize - 1) / blockSize;

    for (int frame = 0; frame < num_frames; frame++) {
        double current_delta = delta * pow(0.98, frame);  // Decrease delta per frame

        // Launch the CUDA kernel for this frame
        computeFractal<<<numBlocks, blockSize>>>(d_pic + frame * height * width, width, height, 1, current_delta);

        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }
    }

    /* end time */
    GET_TIME(end);
    double elapsed = end - start;
    printf("CUDA compute time: %.4f s\n", elapsed);

    /* copy result back to host */
    cudaMemcpy(h_pic, d_pic, num_frames * height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    /* write frames to BMP files */
    if ((width <= 320) && (num_frames <= 100)) { // Avoid large/many files
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &h_pic[frame * height * width], name);
        }
    }
    free(h_pic);
    cudaFree(d_pic);

    return 0;
}
