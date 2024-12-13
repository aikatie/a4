#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"

static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;


__global__ void computeFractal(int width, int height, int num_frames, unsigned char* device_pic) {
    float aspect_ratio = (double)width/height;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int frame = blockIdx.z * blockDim.z + threadIdx.z;

    /* Base Case */
    if (frame > num_frames) {
        return;
    }

    float delta = Delta * pow(0.98, frame);

    const float x0 = xMid - delta * aspect_ratio;
    const float y0 = yMid - delta;
    const float dx = 2.0 * delta * aspect_ratio / width;
    const float dy = 2.0 * delta / height;
    
    if (row < height && col < width) {
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
        } while ((depth > 0) && ((x2 + y2) < 5.0));
        
        device_pic[frame * height * width + row * width + col] = (unsigned char)depth;
      
    }
    
}

int main(int argc, char *argv[]) {
    float start, end;
    
    printf("Fractal v1.6 [parallel]\n");
    
    /* read command line arguments */
    if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
    int width = atoi(argv[1]);
    if (width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
    int height = atoi(argv[2]);
    if (height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
    int num_frames = atoi(argv[3]);
    if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
    
    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);
    
    /* allocate image array */
    int pic_size = num_frames * height * width * sizeof(unsigned char);
    unsigned char *pic = (unsigned char*)malloc(pic_size);
    unsigned char *device_pic;
    
    /* start time */
    cudaMalloc(&device_pic, pic_size);
    cudaDeviceSynchronize();
    GET_TIME(start);
    
    dim3 threads_per_block(32,32,1);
    dim3 num_blocks((width+threads_per_block.x-1)/threads_per_block.x, (height+threads_per_block.y-1)/threads_per_block.y, num_frames);
    
    /* compute frames */
    computeFractal<<<num_blocks, threads_per_block>>>(width, height, num_frames, device_pic);
    
    /* end time */
    cudaDeviceSynchronize();
    GET_TIME(end);

    cudaMemcpy(pic, device_pic, pic_size, cudaMemcpyDeviceToHost);
    cudaFree(device_pic);
    
    cudaError_t cuda_err = cudaGetLastError();
    if (cuda_err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(cuda_err));
    }
    
    double elapsed = end - start;
    printf("Parallel compute time: %.6f s\n", elapsed);

    /* write frames to BMP files */
    if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 2000);
            writeBMP(width, height, &pic[frame * height * width], name);
        }
    }
    
    free(pic);

    return 0;
} /* main */
