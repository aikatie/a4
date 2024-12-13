
#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"

static const double Delta = 0.001; 
static const double xMid =  0.23701;
static const double yMid =  0.521;


__device__ double factor_delta(double first_delta, int frame){
    return first_delta * pow(0.98, frame);
}

__global__ void fractal_kernal(unsigned char *pic, int width, int height, int num_frames, double aspect_ratio){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int frame = blockIdx.z;

    if (col < width && row < height && frame < num_frames){
        double first_delta = Delta;

        const double delta = factor_delta(first_delta, frame);
        const double x0 = xMid - delta * aspect_ratio;
        const double y0 = yMid - delta;
        const double dx = 2.0 * delta * aspect_ratio / width;
        const double dy = 2.0 * delta / height;
        const double cx = x0 + col * dx;
        const double cy = y0 + row * dy;

        double x = cx;
        double y = cy;

        int depth = 256;

        double x2, y2;
        do{
            x2 = x * x;
            y2 = y * y;
            y = 2 * x * y + cy;
            x = x2 - y2 + cx;
            depth--;
        } while ((depth > 0) && ((x2 + y) < 5.0));
        pic[frame * height * width + row * width + col] = (unsigned char)depth;
    }
}


int main(int argc, char *argv[]) {
  double start, end;


  if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
  int height = atoi(argv[2]);
  if (height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
  int num_frames = atoi(argv[3]);
  if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);


  unsigned char *device_pic;
  cudaMalloc((void**)&device_pic, num_frames * height * width * sizeof(unsigned char));
  dim3 th_per_blk(32,32);
  dim3 num_blocks((width + th_per_blk.x - 1) / th_per_blk.x, (height + th_per_blk.y - 1) / th_per_blk.y, num_frames);


  GET_TIME(start);
  const double aspect_ratio = (double) width/height;

  fractal_kernal<<<num_blocks,th_per_blk>>>(device_pic, width, height, num_frames, aspect_ratio);
  cudaDeviceSynchronize();

  GET_TIME(end);


  unsigned char *pic = (unsigned char*)malloc(num_frames * height * width * sizeof(unsigned char));
  cudaMemcpy(pic, device_pic, num_frames * height * width * sizeof(unsigned char),cudaMemcpyDeviceToHost);

  double elapsed = end - start;
  printf("Parallelized compute time: %.6f s\n", elapsed);

  if ((width <= 320) && (num_frames <= 100)) { 
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, height, &pic[frame * height * width], name);
    }
  }
  cudaFree(device_pic);
  free(pic);

  return 0;
} 