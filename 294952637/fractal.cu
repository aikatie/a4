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
#include <cuda.h>
  __global__ void fractalSum(unsigned char *pic, double Delta, int num_frames, double xMid, double yMid, int height, int width) {
          int col = blockIdx.x * blockDim.x + threadIdx.x;
          int row = blockIdx.y * blockDim.y + threadIdx.y;
          int frame = blockIdx.z;
          if(col < width && row < height) {
                const double aspect_ratio = (double)width/height;
                const double delta = Delta * pow(0.98, frame);
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

                pic[frame * height * width + row * width + col] = (unsigned char)depth;

        }

    }
    /*fix loop dependency by using closed-form function */
int main(int argc, char *argv[]) {

  printf("Fractal v1.6 [parallel]\n");
   static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;
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
  unsigned char *pic = (unsigned char*)malloc(num_frames * height * width * sizeof(unsigned char));
  unsigned char *output;
  cudaMalloc((void**) &output, num_frames * width * height * sizeof(unsigned char));
  dim3 th_per_blk(16,16);
  dim3 blk_size((width + th_per_blk.x -1)/th_per_blk.x, (height + th_per_blk.y -1)/th_per_blk.y);
  /* start time */
  double start, end;
  GET_TIME(start);
  /*launch the kernel with th_per_blk */
  fractalSum <<< blk_size,th_per_blk >>> (pic, width, height, xMid, yMid, Delta, num_frames);
  cudaDeviceSynchronize();
  GET_TIME(end);
  double elapsed;
  elapsed = end - start;
  cudaMemcpy(pic, output, num_frames * width * height * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  /* end time */
printf("Parallel compute time: %.6f s\n", elapsed);
  /* write frames to BMP files */
  if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, height, &pic[frame * height * width], name);
    }
  }
  cudaFree(output);
  free(pic);

  return 0;
}
        