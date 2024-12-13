#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "fractal.h"

static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;

__device__ double get_delta(int frame, double aspect_ratio) {
    // Compute delta using a closed-form function to avoid loop dependence
    double delta = Delta * pow(0.98, frame); // delta decreases by 2% per frame
    return delta;
}

__global__ void mandelbrot_kernel(unsigned char *pic, int width, int height, int num_frames, 
                                   double xMid, double yMid, double aspect_ratio) {
    int frame = blockIdx.z;      // Frame index (z-dimension)
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Row index (y-dimension)
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Column index (x-dimension)

    if (row < height && col < width) {
        // Compute delta for the current frame
        double delta = get_delta(frame, aspect_ratio);

        // Coordinates for the current pixel
        const double x0 = xMid - delta * aspect_ratio;
        const double y0 = yMid - delta;
        const double dx = 2.0 * delta * aspect_ratio / width;
        const double dy = 2.0 * delta / height;

        const double cy = y0 + row * dy;
        const double cx = x0 + col * dx;
        
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
        
        // Store the result in the pic array
        pic[frame * height * width + row * width + col] = (unsigned char)depth;
    }
}

int main(int argc, char *argv[]) {
    double start, end;

    printf("Fractal v1.6 [CUDA]\n");

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

    /* allocate image array on host */
    unsigned char *pic = (unsigned char *)malloc(num_frames * height * width * sizeof(unsigned char));

    /* allocate image array on device */
    unsigned char *d_pic;
    cudaMalloc((void**)&d_pic, num_frames * height * width * sizeof(unsigned char));

    /* start time */
    GET_TIME(start);

    const double aspect_ratio = (double)width / height;
    dim3 blockDim(16, 16); // Block size (16x16 threads per block)
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, 
                 (height + blockDim.y - 1) / blockDim.y, 
                 num_frames); // Grid size

    // Compute frames
    for (int frame = 0; frame < num_frames; frame++) {
        // Launch the kernel for each frame
        mandelbrot_kernel<<<gridDim, blockDim>>>(d_pic, width, height, num_frames, xMid, yMid, aspect_ratio);
        
        // Synchronize and check for errors
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
            exit(-1);
        }

        // Copy the result back to host for this frame
        cudaMemcpy(&pic[frame * height * width], d_pic, height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    }

    /* end time */
    GET_TIME(end);
    double elapsed = end - start;
    printf("CUDA compute time: %.4f s\n", elapsed);

    /* write frames to BMP files */
    if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic[frame * height * width], name);
        }
    }

    /* Clean up */
    free(pic);
    cudaFree(d_pic);

    return 0;
}
