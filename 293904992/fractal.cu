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

//DONE BY JULIUS MUHUMUZA AND GIOVANNA SCOZARRO
#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"
#include <math.h>

static const double Delta = 0.001;
static const double xMid =  0.23701;
static const double yMid =  0.521;

__global__ void fractalKernal(unsigned char *pic,double deltaValue,double xMidValue,double yMidValue,int width,int height,double aspect_ratio){
  double myDelta = deltaValue * pow(0.98, blockIdx.x);
  int myFrame = blockIdx.x;
  int myRow = threadIdx.x; 
  int stride = blockDim.x;
 
  const double myX0 = xMidValue - myDelta * aspect_ratio;
  const double myY0 = yMidValue - myDelta;
  const double myDx = 2.0 * myDelta * aspect_ratio / width;
  const double myDy = 2.0 * myDelta / height;
  //const double myCy = myY0 + myRow * myDy;

  for (int i = myRow; i < height; i+=stride){
    const double myCy = myY0 + i * myDy;

    for (int col = 0; col < width; col++) {
      const double myCx = myX0 + col * myDx;
    
    	double x = myCx;
    	double y = myCy;
    	int depth = 256;
    
    	double x2, y2;
    	do {
      		x2 = x * x;
      		y2 = y * y;
      		y = 2 * x * y + myCy;
      		x = x2 - y2 + myCx;
      		depth--;
    	}while ((depth > 0) && ((x2 + y2) < 5.0));
    	pic[myFrame * height * width + i * width + col] = (unsigned char)depth;
      
  	}
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
  unsigned char *pic;// = malloc(num_frames * height * width * sizeof(unsigned char));
  cudaMallocManaged(&pic, (num_frames * height * width * sizeof(unsigned char)));

  /* start time */
  GET_TIME(start);

  /* compute frames */
  const double aspect_ratio = (double)width/height;

  int numThreads = (height <= 1024) ? height:1024;
  int numBlocks = num_frames;
  fractalKernal<<<numBlocks, numThreads>>>(pic,Delta,xMid,yMid,width,height,aspect_ratio);
  cudaDeviceSynchronize();
  
  /* end time */
  GET_TIME(end);
  double elapsed = end - start;
  printf("Parallel compute time: %.6f s\n", elapsed);

  /* write frames to BMP files */
  if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
    for (int frame = 0; frame < num_frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, height, &pic[frame * height * width], name);
    }
  }
  cudaFree(pic);
  //free(pic);

  return 0;
} /* main */