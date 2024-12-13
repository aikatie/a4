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
#include <cuda_runtime.h>
#include <math.h>

static const double Delta = 0.001;
static const double xMid = 0.23701;
static const double yMid = 0.521;

__global__ void computeFractal(unsigned char *imageData, int imgWidth, int imgHeight, int frameCount, double aspectRatio) {
  int pixelRow = blockIdx.y * blockDim.y + threadIdx.y;
  int pixelCol = blockIdx.x * blockDim.x + threadIdx.x;
  int frameIndex = blockIdx.z;

  if (pixelRow < imgHeight && pixelCol < imgWidth) {
    const double zoomFactor = Delta * pow(.98, frameIndex);
    const double startX = xMid - zoomFactor * aspectRatio;
    const double startY = yMid - zoomFactor;
    const double deltaX = 2.0 * zoomFactor * aspectRatio / imgWidth;
    const double deltaY = 2.0 * zoomFactor / imgHeight;

    const double complexY = startY + pixelRow * deltaY;
    const double complexX = startX + pixelCol * deltaX;

    double realPart = complexX;
    double imaginaryPart = complexY;
    int pixelDepth = 256;
    double realSquare, imaginarySquare;

    do {
      realSquare = realPart * realPart;
      imaginarySquare = imaginaryPart * imaginaryPart;
      imaginaryPart = 2 * realPart * imaginaryPart + complexY;
      realPart = realSquare - imaginarySquare + complexX;
      pixelDepth--;
    } while ((pixelDepth > 0) && ((realSquare + imaginarySquare) < 5.0));
    imageData[frameIndex * imgHeight * imgWidth + pixelRow * imgWidth + pixelCol] = (unsigned char)pixelDepth;
  }
}

int main(int argc, char *argv[]) {
  double startTime, endTime;
  unsigned char *deviceImageData;
  printf("Fractal v1.6 [CUDA]\n");

  /* read command line arguments */
  if (argc != 4) {fprintf(stderr, "usage: %s height width num_frames\n", argv[0]); exit(-1);}
  int imgWidth = atoi(argv[1]);
  if (imgWidth < 10) {fprintf(stderr, "error: width must be at least 10\n"); exit(-1);}
  int imgHeight = atoi(argv[2]);
  if (imgHeight < 10) {fprintf(stderr, "error: height must be at least 10\n"); exit(-1);}
  int frameCount = atoi(argv[3]);
  if (frameCount < 1) {fprintf(stderr, "error: num_frames must be at least 1\n"); exit(-1);}
  printf("Computing %d frames of %d by %d fractal\n", frameCount, imgWidth, imgHeight);

  /* allocate image array */
  unsigned char *imageData = (unsigned char *)malloc(frameCount * imgHeight * imgWidth * sizeof(unsigned char));
  cudaMalloc(&deviceImageData, frameCount * imgHeight * imgWidth * sizeof(unsigned char));

  if (imageData == NULL){
    fprintf(stderr, "Error: Memory allocation failed\n");
    exit(-1);
  }

  cudaError_t cudaStatus = cudaMalloc(&deviceImageData, frameCount * imgHeight * imgWidth * sizeof(unsigned char));
  if (cudaStatus != cudaSuccess) {
    fprintf(stderr, "error: cudaMalloc failed\n");
    exit(-1);
  }

  /* start time */
  GET_TIME(startTime);

  /* compute frames */
  const double aspectRatio = (double)imgWidth / imgHeight;
  dim3 threadsPerBlock(16, 16, 1);
  dim3 numBlocks((imgWidth + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                 (imgHeight + threadsPerBlock.y - 1) / threadsPerBlock.y, 
                 frameCount);
  computeFractal<<<numBlocks, threadsPerBlock>>>(deviceImageData, imgWidth, imgHeight, frameCount, aspectRatio);
  cudaMemcpy(imageData, deviceImageData, frameCount * imgHeight * imgWidth * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  /* end time */
  GET_TIME(endTime);
  double elapsedTime = endTime - startTime;
  printf("CUDA compute time: %.4f s\n", elapsedTime);

  /* write frames to BMP files */
  if ((imgWidth <= 320) && (frameCount <= 100)) { /* do not write if images large or many */
    for (int frameIndex = 0; frameIndex < frameCount; frameIndex++) {
      char fileName[32];
      sprintf(fileName, "fractal%d.bmp", frameIndex + 1000);
      writeBMP(imgWidth, imgHeight, &imageData[frameIndex * imgHeight * imgWidth], fileName);
    }
  }

  free(imageData);
  cudaFree(deviceImageData);

  return 0;
} /* main */
