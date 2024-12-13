#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "fractal.h"
#include "timer.h"

static const double Delta = 0.001;
static const double xMid = 0.23701;
static const double yMid = 0.521;

__global__ void fractalKernel(unsigned char *pic, int width, int height, double delta_start, int num_frames) {
    int pixel_index = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height * num_frames;
    if (pixel_index >= total_pixels) return;
    int frame = pixel_index / (width * height);
    int frame_pixel = pixel_index % (width * height);
    int row = frame_pixel / width;
    int col = frame_pixel % width;
    double delta = delta_start * pow(0.98, frame);
    double aspect_ratio = (double)width / height;
    double x0 = xMid - delta * aspect_ratio;
    double y0 = yMid - delta;
    double dx = 2.0 * delta * aspect_ratio / width;
    double dy = 2.0 * delta / height;
    double cx = x0 + col * dx;
    double cy = y0 + row * dy;
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
    pic[pixel_index] = (unsigned char)depth;
}

int main(int argc, char *argv[]) {
    double start, end;
    printf("Fractal v1.6 [CUDA]\n");
    if (argc != 4) { fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1); }
    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int num_frames = atoi(argv[3]);
    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);
    unsigned char *pic, *d_pic;
    size_t total_pixels = num_frames * height * width;
    size_t size = total_pixels * sizeof(unsigned char);
    pic = (unsigned char *)malloc(size);
    cudaMalloc(&d_pic, size);
    cudaMemset(d_pic, 0, size);
    GET_TIME(start);
    int th_per_blk = 256;
    int num_blks = (total_pixels + th_per_blk - 1) / th_per_blk;
    fractalKernel<<<num_blks, th_per_blk>>>(d_pic, width, height, Delta, num_frames);
    cudaDeviceSynchronize();
    cudaMemcpy(pic, d_pic, size, cudaMemcpyDeviceToHost);
    GET_TIME(end);
    double elapsed = end - start;
    printf("CUDA compute time: %.6f s\n", elapsed);
    if ((width <= 320) && (num_frames <= 100)) {
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic[frame * height * width], name);
        }
    }
    free(pic);
    cudaFree(d_pic);
    return 0;
}
