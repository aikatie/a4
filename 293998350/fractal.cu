// Worked as a pair
// Collaborator: Julius Muhumuza, juliusnm@udel.edu

#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"
#include <math.h>

// Given variables
static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;

__global__ void ComputeFractal(int width, int height, int num_frames, double aspect_ratio, unsigned char* pic) {

  int frame = blockIdx.x;
  int row = threadIdx.x;
  int stride = blockDim.x;
  // serial program multiplied delta by .98 every iteration
  double frameDelta = Delta * pow(0.98, frame);
  
  const double x0 = xMid - frameDelta * aspect_ratio;
  const double y0 = yMid - frameDelta;
  const double dx = 2.0 * frameDelta * aspect_ratio / width;
  const double dy = 2.0 * frameDelta / height;

  for (int i = row; i < height; i += stride){
    double cy = y0 + i * dy;

    for (int col = 0; col < width; col++) {
      double cx = x0 + col * dx;

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
          
      pic[frame * height * width + i * width + col] = (unsigned char)depth;
    }
  }
}
  
int main(int argc, char *argv[]) {
  double start, end;

  printf("Fractal v1.6 [{parallel}]\n");

  /* read command line arguments */
  if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
  int height = atoi(argv[2]);
  if (height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
  int num_frames = atoi(argv[3]);
  if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

  /* allocate image array using CUDA */
  unsigned char *pic;
  cudaMallocManaged(&pic, (num_frames * height * width * sizeof(unsigned char)));

  GET_TIME(start);

  /* compute frames (same for every block) */
  const double aspect_ratio = (double)width/height;
  
  // no more than 1024 threads per block
  int numThreads = (height <= 1024) ? height:1024;
  int numBlocks= num_frames;

  ComputeFractal<<<numBlocks, numThreads>>>(width, height, num_frames, aspect_ratio, pic);
  cudaDeviceSynchronize(); // blocks CPU until GPU finished 

  GET_TIME(end);

  double elapsed = end - start;
  printf("Parallel compute time: %.4f s\n", elapsed);

  /* write frames to BMP files */
  if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, height, &pic[frame * height * width], name);
    }
  }
  cudaFree(pic); // free cuda-allocated array

  return 0;
} /* main */