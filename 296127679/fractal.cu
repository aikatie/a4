#include <stdlib.h>   
#include <stdio.h>                                                          
#include "timer.h"
#include "fractal.h"
#include <cuda_runtime.h>

__global__ void fractal(unsigned char *pic, int width, int height, int num_frames, double xMid, double yMid, double delta){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = blockIdx.z;

    if (row < height && col < width){
    double aspect_ratio = (double)width / height;
    double x0 = xMid - delta * aspect_ratio;
    double y0 = yMid - delta;
    double dx = 2.0 * delta * aspect_ratio / width;
    double dy = 2.0 * delta / height;

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

static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;

int main(int argc, char *argv[]) {
  double start, end;

  printf("Fractal v1.6 [Cuda]\n");

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
  unsigned char *pic;
  cudaMalloc((void **)&pic, num_frames * height * width * sizeof(unsigned char));
  unsigned char *pic2 = (unsigned char *)malloc(num_frames * height * width * sizeof(unsigned char));
  /* start time */
  GET_TIME(start);

  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((width + threadsPerBlock.x -1)/threadsPerBlock.x, (height + threadsPerBlock.y)/threadsPerBlock.y, num_frames);

  fractal<<<numBlocks,threadsPerBlock>>>(pic, width, height, num_frames, xMid, yMid, Delta);
  cudaDeviceSynchronize();
  cudaMemcpy(pic, pic2, num_frames * height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);


  /* end time */
  GET_TIME(end);
  double elapsed = end - start;
  printf("Cuda compute time: %.6f s\n", elapsed);

  /* write frames to BMP files */
  if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, height, &pic2[frame * height * width], name);
    }
  }
  free(pic2);
  cudaFree(pic);

  return 0;
} /* main */                  
 