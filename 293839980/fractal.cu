/*
Computing a movie of zooming into a fractal

Original C++ code by Martin Burtscher, Texas State University

Reference: E. Ayguade et al., 
           "Peachy Parallel Assignments (EduHPC 2018)".
           2018 IEEE/ACM Workshop on Education for High-Performance Computing (EduHPC), pp. 78-85,
           doi: 10.1109/EduHPC.2018.00012

Copyright (c) 2018, Texas State University. All rights reserved.

Redistribution and usage in source and binary form, with or without
modification, is only permitted for educational use.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

/* 
Neil Irungu
Kareena Keswani
*/
#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"

static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;

__global__ void computeFractalKernel (unsigned char* pic, int width, int height, int num_frames, double Delta, double xMid, double yMid) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height * num_frames;
    int area = width * height; 

    while (idx < total_pixels){
	int frame = idx / area;
	int remainder = idx % area;
	int row = remainder / width;
	int col = remainder % width;

	double delta = Delta * pow(0.98, frame);

	double aspect_ratio = (double) width / height;
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

	pic[idx] = (unsigned char) depth;

	idx += blockDim.x * gridDim.x;

    }

}


int main(int argc, char *argv[]) {
  double start, end;

  printf("Fractal v1.6 [CUDA]\n");

  /* read command line arguments */
  if (argc != 5) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
  int height = atoi(argv[2]);
  if (height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
  int num_frames = atoi(argv[3]);
  if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  int th_per_blk = atoi(argv[4]);
  if (th_per_blk < 1) {fprintf(stderr, "error: th_per_blk must be at least 1\n"); exit(-1);}
  printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

  /* allocate image array */
  size_t total_size = num_frames * height * width * sizeof(unsigned char);
  unsigned char *pic = (unsigned char*)malloc(total_size);

  unsigned char *d_pic;
  cudaMalloc(&d_pic, total_size);

  int total_pixels = width * height * num_frames;
  int num_blocks = (total_pixels + th_per_blk - 1) / th_per_blk;

  /* start time */
  GET_TIME(start);

  computeFractalKernel <<<num_blocks, th_per_blk>>>(d_pic, width, height, num_frames, Delta, xMid, yMid);

  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess){
    fprintf(stderr, "CUDA ERROR: %s\n", cudaGetErrorString(err));
    exit(-1);
  }

  cudaMemcpy(pic, d_pic, total_size, cudaMemcpyDeviceToHost);


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

  free(pic);
  cudaFree(d_pic);

  return 0;
} /* main */
