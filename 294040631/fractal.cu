// Sophia Romero and Marc Madlangbayan
/*                                                                          Computing a movie of zooming into a fractal                                                                                                             Original C++ code by Martin Burtscher, Texas State University                                                                                           Reference: E. Ayguade et al.,                                                          "Peachy Parallel Assignments (EduHPC 2018)".                                2018 IEEE/ACM Workshop on Education for High-Performance Computing (EduHPC), pp. 78-85,
           doi: 10.1109/EduHPC.2018.00012

Copyright (c) 2018, Texas State University. All rights reserved.
                                                                            Redistribution and usage in source and binary form, with or without
modification, is only permitted for educational use.                                                                                                    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND                                                                         ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED                                                                           WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE      DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON                                                                          ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT     (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS                                                                           SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.                                                                                            Author: Martin Burtscher                                                    */ 

#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"
#include <math.h>

static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;

__global__ void computeFractal(unsigned char *pic, int width, int height, int num_frames, double delta_initial, double zoom_factor, double xMid, double yMid) {
  int gidx = blockIdx.x * blockDim.x + threadIdx.x;

  int frame = gidx / (width * height);
  int pixel_index = gidx % (width * height);

  int row = pixel_index / width;
  int col = pixel_index % width;
  /* compute frames */
  // computes delta for single frame vv
  double delta = delta_initial * pow(zoom_factor, frame);
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

   pic[gidx] = (unsigned char)depth;
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

   /* allocate image array */
  unsigned char *pic = (unsigned char *)malloc(num_frames * height * width * sizeof(unsigned char));

  unsigned char *d_pic;
  cudaMalloc(&d_pic, num_frames * height * width * sizeof(unsigned char));

  int th_per_blk = 256;
  int total_pixel = width * height * num_frames;
  int block_per_grid = (total_pixel + th_per_blk - 1) / th_per_blk;

  /* start time */
  GET_TIME(start);

  computeFractal<<<block_per_grid, th_per_blk>>>(d_pic, width, height, num_frames, Delta, 0.98, xMid, yMid);
  cudaDeviceSynchronize();

  /* end time */
  GET_TIME(end);
  double elapsed = end - start;
  printf("CUDA compute time: %.4f s\n", elapsed);

  cudaMemcpy(pic, d_pic, num_frames * height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  /* write frames to BMP files */
  if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, height, &pic[frame * height * width], name);
    }
  }

  cudaFree(d_pic);
  free(pic);

  return 0;
} /* main */

