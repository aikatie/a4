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
#include <cuda_runtime.h>

static const float Delta = 0.001;
static const float xMid =  0.23701;
static const float yMid =  0.521;

void computeFractalSerial(int height, int width, int num_frames, float aspect_ratio, unsigned char * pic){
  float delta = Delta;
  for (int frame = 0; frame < num_frames; frame++) {

    const float x0 = xMid - delta * aspect_ratio;
    const float y0 = yMid - delta;
    const float dx = 2.0 * delta * aspect_ratio / width;
    const float dy = 2.0 * delta / height;
    
    for (int row = 0; row < height; row++) {

      const float cy = y0 + row * dy;
      
      for (int col = 0; col < width; col++) {

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
        
        pic[frame * height * width + row * width + col] = (unsigned char)depth;
      
      }
    }
    delta *= 0.98;
  }
}

__host__ __device__ float calculateDelta(int offset) {
  float delta = Delta;
  for (int i = 0; i < offset; i++) {
	  delta *= 0.98;
  }
  return delta;
}

__global__ void computeFractalParallel(int frames, int height, int width, float aspect_ratio, unsigned char * pic) {
  
  int row = ((blockDim.y) * blockIdx.y) + threadIdx.y;
  int col = ((blockDim.x) * blockIdx.x) + threadIdx.x;
  int frame = blockIdx.z;

  //printf("row: %d, col: %d, frame: %d\n", row, col, frame);
  
  // outer most for loop
  float delta = calculateDelta(frame);
  const float x0 = xMid - delta * aspect_ratio;
  const float y0 = yMid - delta;
  const float dx = 2.0 * delta * aspect_ratio / width;
  const float dy = 2.0 * delta / height;

  // middle most for loop
  const float cy = y0 + row * dy;

  // inner most for loop
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
 
  if (frame < frames && row < height && col < width) {
	// frame * height * width + row * width + col
	int index = frame * height * width + row * width + col;
        //printf("Editing Index: %d\n", index);
  	pic[index] = (unsigned char) depth;
  } 
}


int main(int argc, char *argv[]) {
  double start, end;

  printf("Fractal v1.6 [serial]\n");

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
  unsigned char *d_pic;
  unsigned char *pic = (unsigned char * ) malloc(num_frames * height * width * sizeof(unsigned char)); 
  cudaMallocManaged(&d_pic, num_frames * height * width * sizeof(unsigned char));
  const float aspect_ratio = (float)width/height;
  
  // Parallel Program Run time
  dim3 blockDim(32, 32, 1);
  dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1)/ blockDim.y, num_frames);
   
  GET_TIME(start);
  computeFractalParallel <<<gridDim, blockDim>>> (num_frames, height, width, aspect_ratio, d_pic);
  cudaDeviceSynchronize();
  GET_TIME(end);
  printf("Parallel compute time: %.6f s\n", end - start);
  cudaMemcpy(pic, d_pic, height * width * num_frames * sizeof(unsigned char), cudaMemcpyDeviceToHost);

  // Serial Program Run time 
  /* start time */
  GET_TIME(start);
  // /* compute frames */
  computeFractalSerial(height, width, num_frames, aspect_ratio, pic);
  // /* end time */
  GET_TIME(end);
  double elapsed = end - start;
  printf("Serial compute time: %.6f s\n", elapsed);

  /* write frames to BMP files */
  if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000); 
      writeBMP(width, height, &pic[frame * height * width], name);
    }
  }
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA Error: %s\n", cudaGetErrorString(err));
}
  cudaFree(d_pic);
  free(pic);
  return 0;
} /* main */
