#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"

static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;

/* CUDA kernel to compute a single frame of the fractal */
__global__ void computeFractal(unsigned char* pic, int width, int height, double Delta, double xMid, double yMid) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int frame = blockIdx.z;

    double delta = Delta * pow(0.98, frame);

    if (col < width && row < height) {
        const double aspect_ratio = (double) width / height;
        const double x0 = xMid - delta * aspect_ratio;
        const double y0 = yMid - delta;
        const double dx = 2.0f * delta * aspect_ratio / width;
        const double dy = 2.0f * delta / height;

        const double cx = x0 + col * dx;
        const double cy = y0 + row * dy;

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

    printf("Fractal v1.6 [CUDA]\n");

    /* read command line arguments */
    if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
    int width = atoi(argv[1]);
    if (width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
    int height = atoi(argv[2]);
    if (height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
    int num_frames = atoi(argv[3]);
    if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

    /* allocate image array using unified memory */
     unsigned char* pic = (unsigned char*)malloc(num_frames * height * width * sizeof(unsigned char));
     unsigned char* d_pic;
     cudaMalloc(&d_pic, num_frames * height * width * sizeof(unsigned char));

    /* start time */
    GET_TIME(start);

    /* define CUDA grid and block sizes */
    dim3 th_per_blk(32, 32); // Increase threads per block for higher occupancy
    dim3 numBlocks((width + th_per_blk.x - 1) / th_per_blk.x, (height + th_per_blk.y - 1) / th_per_blk.y, num_frames);

    /* launch CUDA kernel */
    computeFractal<<<numBlocks, th_per_blk>>>(d_pic, width, height, Delta, xMid, yMid);
    cudaDeviceSynchronize(); // Ensure kernel completion before the next iteration
    cudaMemcpy(pic, d_pic,num_frames * height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    /* end time */
    GET_TIME(end);
    double elapsed = end - start;
    printf("CUDA compute time: %.6f s\n", elapsed);

    /* write frames to BMP files */
    if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic[frame * height * width], name);
        }
    }

    /* free memory */
    free(pic);
    cudaFree(d_pic);

    return 0;
} /* main */

// Group: Dylan Blevins and Colin Barry
