#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"
#include <math.h>

static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;
__global__ void fractal(int num_frames, int width, int height, unsigned char* pic);
int main(int argc, char *argv[]) {
  double start, end;

  printf("Fractal v1.6 [serial]\n");

  if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
  int height = atoi(argv[2]);
  if (height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
  int num_frames = atoi(argv[3]);
  if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

  unsigned char *pic; 
  cudaMallocManaged(&pic, num_frames * height * width * sizeof(unsigned char));

  GET_TIME(start);

  int th_per_block = 1024;
  int N = num_frames * height * width;
  int blk_ct = (N + th_per_block - 1) / th_per_block;
  fractal <<<blk_ct,th_per_block>>> (num_frames, width, height, pic);
  cudaDeviceSynchronize();
  const double aspect_ratio = (double)width/height;
  double delta = Delta;

  GET_TIME(end);
  double elapsed = end - start;
  printf("Serial compute time: %.6f s\n", elapsed);

  if ((width <= 320) && (num_frames <= 100)) {
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, height, &pic[frame * height * width], name);
    }
  }

  cudaFree(pic);

  return 0;
}

__global__ void fractal(int num_frames, int width, int height, unsigned char* pic){
  const double aspect_ratio = (double)width/height;
  double delta = Delta;
  int threadx = blockIdx.x * blockDim.x + threadIdx.x;
  int frame = threadx/(height*width);
  int row = (threadx/width) % height;
  int col = threadx % width;

  if(frame < num_frames){
    delta = pow(0.98,frame); 
    const double x0 = xMid - delta * aspect_ratio;
    const double y0 = yMid - delta;
    const double dx = 2.0 * delta * aspect_ratio / width;
    const double dy = 2.0 * delta / height;

    if(row < height){
      const double cy = y0 + row * dy;
      if(col < width){
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
        pic[frame * height * width + row * width + col] = (unsigned char)depth;
      }
    }
  }
}
