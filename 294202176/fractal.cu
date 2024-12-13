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

__global__ void fractal_Kernel(int width, int height, int num_frames, unsigned char *pic_array ){
    float aspect_ratio = (float)width / (float)height;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x +threadIdx.x;
    int frame_index = blockIdx.z * blockDim.z +threadIdx.z;
    if (frame_index > num_frames){
      return;
    }
        float delta = Delta * pow(0.98, frame_index);

        const float x0 = xMid - delta * aspect_ratio;
        const float y0 = yMid - delta;
        const float dx = 2.0 * delta * aspect_ratio / width;
        const float dy = 2.0 * delta / height;

    if (row < height && col< width){
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

        pic_array[frame_index * height * width + row * width + col] = (unsigned char)depth;

      }
  }

int main(int argc, char *argv[]) {
  double start, end;

  printf("Fractal v1.6 [Parallel]\n");

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
  int picSize = (sizeof(unsigned char) * num_frames * height* width);
  unsigned char *device_pic;
  unsigned char *host_pic= (unsigned char *)malloc(picSize);

  /* start time */
  cudaMalloc(&device_pic,picSize);

  cudaDeviceSynchronize();
  GET_TIME(start); 
  dim3 threads_per_block(32,32,1);
  dim3 num_blocks((width+threads_per_block.x-1)/threads_per_block.x, (height+threads_per_block.y-1)/threads_per_block.y,num_frames);
  fractal_Kernel<<<num_blocks,threads_per_block>>>(width,height,num_frames, device_pic);
  cudaDeviceSynchronize();
  GET_TIME(end)

  cudaMemcpy(host_pic, device_pic, picSize, cudaMemcpyDeviceToHost);
  cudaFree(device_pic);

  cudaError_t cuda_err = cudaGetLastError();
  if (cuda_err != cudaSuccess){
    fprintf(stderr, "CUDA erorr: %s\n", cudaGetErrorString(cuda_err));
  }

  float elapsed = end - start;
  printf("Parallel compute time: %.6f s\n", elapsed);

  /* write frames to BMP files */
  if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, height, &host_pic[frame * height * width], name);
    }
  }
  //cudaFree(device_pic);
  free(host_pic);
  return 0;
} /* main */
