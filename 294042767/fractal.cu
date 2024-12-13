/*
Computing a movie of zooming into a fractal using Cuda

Author: Jonathan Ma
*/

#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"


__constant__ double Delta;
__constant__ double xMid;
__constant__ double yMid;
__constant__ int height;
__constant__ int width;
__constant__ double aspect_ratio;


__global__ 
void cuda_hello(){
        printf("Hello World from GPU!\n");
}

__global__ 
void compute_fractal(unsigned char *pic, int num_frames) {
  extern __shared__ unsigned char shared_frame[];
  int frame = blockIdx.z;
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= height || col >= width || frame >= num_frames) return;
  const double delta = Delta*pow(0.98, frame);
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
  int local_index = threadIdx.y * blockDim.x + threadIdx.x;
  shared_frame[local_index] = (unsigned char)depth;
  __syncthreads();
  if (row < height && col < width) {
      pic[frame * height * width + row * width + col] = shared_frame[local_index];
  }
}

int main(int argc, char *argv[]) {
  double start, end;

  printf("Fractal v1.0 [cuda]\n");

  /* read command line arguments */
  if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
  int hst_width = atoi(argv[1]);
  if (hst_width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
  int hst_height = atoi(argv[2]);
  if (hst_height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
  int num_frames = atoi(argv[3]);
  if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  printf("Computing %d frames of %d by %d fractal\n", num_frames, hst_width, hst_height);

  /* allocate image array */
  int sz = num_frames * hst_height * hst_width * sizeof(unsigned char);

  unsigned char *pic = (unsigned char*)malloc(sz);
  unsigned char *dev_pic;

  int malloc_status = cudaMalloc((void**) &dev_pic, sz);
  if(malloc_status == 0){
      printf("Successfully allocated device memory.\n");
  }
  if(malloc_status == 100){
      printf("No Cuda capable devices found.\n");
  }

  /* start time */
  GET_TIME(start);

  /* compute frames */
  const double hst_aspect_ratio = (double)hst_width/hst_height;

  cudaMemcpy(dev_pic, pic, sz, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(aspect_ratio, &hst_aspect_ratio, sizeof(double));
  cudaMemcpyToSymbol(height, &hst_height, sizeof(int));
  cudaMemcpyToSymbol(width, &hst_width, sizeof(int));

  static const double hst_Delta = 0.001;
  static const double hst_xMid =  0.23701;
  static const double hst_yMid =  0.521;

  cudaMemcpyToSymbol(Delta, &hst_Delta, sizeof(double));
  cudaMemcpyToSymbol(xMid, &hst_xMid, sizeof(double));
  cudaMemcpyToSymbol(yMid, &hst_yMid, sizeof(double));


  dim3 blockSize(32,32);
  dim3 gridSize((hst_width + blockSize.x - 1) / blockSize.x,
          (hst_height + blockSize.y - 1) / blockSize.y, 
          num_frames);
  int sharedMemSize = blockSize.x * blockSize.y * sizeof(unsigned char);
  compute_fractal<<<gridSize, blockSize, sharedMemSize>>>(dev_pic, num_frames);
  cudaDeviceSynchronize();
  cudaMemcpy(pic, dev_pic, sz, cudaMemcpyDeviceToHost);

  /* end time */
  GET_TIME(end);
  double elapsed = end - start;
  printf("Cuda compute time: %.4f s\n", elapsed);
  /* write frames to BMP files */
  if ((hst_width <= 320) && (num_frames <= 10)) { /* do not write if images large or many */
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(hst_width, hst_height, &pic[frame * hst_height * hst_width], name);
    }
  }

  cudaFree(dev_pic);
  free(pic);
  return 0;
} /* main */
