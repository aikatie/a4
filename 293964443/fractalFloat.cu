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
#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"

static const float Delta = 0.001;
static const float xMid =  0.23701;
static const float yMid =  0.521;

__device__ float computeDelta(int frame, int num_frames) {
    return Delta * pow(0.98, frame);
}

__global__ void computeFractal(unsigned char *pic, int width, int height, int num_frames) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int frame = idx / (height * width);
    int row = (idx / width) % height;
    int col = idx % width;

    const float aspect_ratio = (float)width / height;
    const float delta = computeDelta(frame, num_frames);

    const float x0 = xMid - delta * aspect_ratio;
    const float y0 = yMid - delta;
    const float dx = 2.0 * delta * aspect_ratio / width;
    const float dy = 2.0 * delta / height;

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

    pic[idx] = (unsigned char)depth;
}

int main(int argc, char *argv[]) {
  double start, end;

  printf("Fractal v1.6 [serial]\n");

  // Command Line Arguments
  if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
  int width = atoi(argv[1]);
  if (width < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
  int height = atoi(argv[2]);
  if (height < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
  int num_frames = atoi(argv[3]);
  if (num_frames < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

  // Image Array Allocation
  unsigned char *pic;
  cudaMallocManaged(&pic, num_frames * height * width * sizeof(unsigned char));

  // Start Time
  GET_TIME(start);

  // Compute the Frames
  int num_pixels = width * height * num_frames;
    int th_per_blk = 256;
    int num_blks = (num_pixels + th_per_blk - 1) / th_per_blk;
    computeFractal<<<num_blks, th_per_blk>>>(pic, width, height, num_frames);
     cudaDeviceSynchronize();

  // End Time
  GET_TIME(end);
  double elapsed = end - start;
  printf("Cuda compute time: %.4f s\n", elapsed);

  // Frames to BMP Files
  if ((width <= 320) && (num_frames <= 100)) { // Not if image in too large
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, height, &pic[frame * height * width], name);
    }
  }

  cudaFree(pic);

  return 0;
} /* main */