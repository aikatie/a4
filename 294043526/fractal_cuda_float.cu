#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"

static const float Delta = 0.001f;
static const float xMid =  0.23701f;
static const float yMid =  0.521f;

__global__ void computeFractal(int width, int height, float delta, float aspect_ratio, unsigned char *pic, int frame) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        const float x0 = xMid - delta * aspect_ratio;
        const float y0 = yMid - delta;
        const float dx = 2.0f * delta * aspect_ratio / width;
        const float dy = 2.0f * delta / height;
        
        const float cy = y0 + row * dy;
        const float cx = x0 + col * dx;
        
        float x = cx;
        float y = cy;
        int depth = 256;
        
        float x2, y2;
        do {
            x2 = x * x;
            y2 = y * y;
            y = 2 * x * y + cy;
            x = x2 - y2 + cx;
            depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0f));
        
        pic[frame * height * width + row * width + col] = (unsigned char)depth;
    }
}

int main(int argc, char *argv[]) {
    double start, end;
    
    printf("Fractal v1.6 [CUDA] - Using floats for computation and doubles for time tracking\n");

    /* read command line arguments */
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
    unsigned char *pic;
    cudaMallocManaged(&pic, num_frames * height * width * sizeof(unsigned char));

    /* start time */
    GET_TIME(start);

    /* compute frames */
    const float aspect_ratio = (float)width / height;
    float delta = Delta;
        dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((width + 15) / 16, (height + 15) / 16);
    
    for (int frame = 0; frame < num_frames; frame++) {
        computeFractal<<<numBlocks, threadsPerBlock>>>(width, height, delta, aspect_ratio, pic, frame);
        
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(error));
            return -1;
        }

        delta *= 0.98f;
    }

    /* end time */
    GET_TIME(end);
    double elapsed = end - start;
    printf("CUDA compute time: %.6f s\n", elapsed);  // Using double precision for time

    /* write frames to BMP files */
    if ((width <= 320) && (num_frames <= 100)) {
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic[frame * height * width], name);
        }
    }

    cudaFree(pic);
    
    return 0;
} /* main */
