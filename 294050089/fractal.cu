#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"
#include <cuda_runtime.h>

static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;


__global__ void computeFractal(unsigned char *pic, int width, int height, int num_frames, double delta, double aspect_ratio) {
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int frame = blockIdx.z * blockDim.z + threadIdx.z; // Frame index

    if (col < width && row < height && frame < num_frames) {
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
       pic[frame * height * width + row * width + col] = (unsigned char)depth;
        }
}

int main(int argc, char *argv[]) {
    double start, end;
    printf("Fractal v1.6 [CUDA]\n");
    if (argc != 4) {                   
    fprintf(stderr, "usage: %s height width num_frames\n", argv[0]);
                    exit(-1);
        }
    int width = atoi(argv[1]);
    if (width < 10) { fprintf(stderr, "error: width must be at least 10\n"); exit(-1); }
    int height = atoi(argv[2]);
    if (height < 10) { fprintf(stderr, "error: height must be at least 10\n"); exit(-1); }
    int num_frames = atoi(argv[3]);
    if (num_frames < 1) { fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1); }
    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);
    unsigned char *pic = (unsigned char*)malloc(num_frames * height * width * sizeof(unsigned char));
    unsigned char *d_pic;
    cudaMalloc((void **)&d_pic, num_frames * height * width * sizeof(unsigned char));
    double aspect_ratio = (double)width / height;
    double delta = Delta;
    dim3 threadsPerBlock(16, 16, 1);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16, num_frames);
    GET_TIME(start);
    for (int frame = 0; frame < num_frames; frame++) {
        computeFractal<<<numBlocks, threadsPerBlock>>>(d_pic, width, height, num_frames, delta, aspect_ratio);
        delta *= 0.98;
    }
    cudaMemcpy(pic, d_pic, num_frames * height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    cudaFree(d_pic);
    GET_TIME(end);
    double elapsed = end - start;
    printf("CUDA compute time: %.4f s\n", elapsed);
    if ((width <= 320) && (num_frames <= 100)) {
            for (int frame = 0; frame < num_frames; frame++) {
                    char name[32];
                    sprintf(name, "fractal%d.bmp", frame + 1000);
                    writeBMP(width, height, &pic[frame * height * width], name);
                }
        }
  free(pic);
  return 0;
}