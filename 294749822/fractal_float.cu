#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"
#include <cuda_runtime.h>

static const float Delta = 0.001;
static const float xMid = 0.23701;
static const float yMid = 0.521;

// CUDA kernel for computing fractal
__global__ void computeFractal(unsigned char* pic, int width, int height, int num_frames,
                              float x0_first, float y0_first, float aspect_ratio) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int frame = blockIdx.z;


    if (col < width && row < height && frame < num_frames) {
        float delta = Delta * pow(0.98, frame);
        
        float x0 = xMid - delta * aspect_ratio;
        float y0 = yMid - delta;
        float dx = 2.0 * delta * aspect_ratio / width;
        float dy = 2.0 * delta / height;

        float cx = x0 + col * dx;
        float cy = y0 + row * dy;
        int depth = 256;
        float x2, y2;

        float x = cx;
        float y = cy; 
        
        
        do {
            x2 = x * x;
            y2 = y * y;
            y = 2 * x * y + cy;
            x = x2 - y2 + cx;
            depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));
    
    // Store result
        pic[frame * height * width + row * width + col] = (unsigned char)depth;
    }
}

int main(int argc, char *argv[]) {
    double start, end;
    
    printf("Fractal v1.6 [CUDA]\n");
    
    // Check command line arguments
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
    
    // Allocate host and device memory
    unsigned char *h_pic = (unsigned char*)malloc(num_frames * height * width * sizeof(unsigned char));
    unsigned char *d_pic;
    cudaMalloc(&d_pic, num_frames * height * width * sizeof(unsigned char));
    
    // Calculate initial values
    const float aspect_ratio = (float)width / height;
    const float x0_first = xMid - Delta * aspect_ratio;
    const float y0_first = yMid - Delta;
    
    dim3 threadsPerBlock(16, 16);  // 256 threads per block
    dim3 numBlocks((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (height + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   num_frames
    );
    
    GET_TIME(start);
    
    // Launch kernel
    computeFractal<<<numBlocks, threadsPerBlock>>>(d_pic, width, height, num_frames, x0_first, y0_first, aspect_ratio);
    
    
    GET_TIME(end);

    cudaMemcpy(h_pic, d_pic, num_frames * height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    double elapsed = end - start;
    printf("CUDA compute time: %.6f s\n", elapsed);
    
    // Write frames to BMP files
    if ((width <= 1024) && (num_frames <= 100)) {
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "./fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &h_pic[frame * height * width], name);
        }
    }
    
    // Free memory
    free(h_pic);
    cudaFree(d_pic);
    
    return 0;
}