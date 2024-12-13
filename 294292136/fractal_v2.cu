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

static const float Delta = 0.001f;
static const float xMid =  0.23701f;
static const float yMid =  0.521f;

__global__ void fractal(unsigned char *pic, int width, int height,  int num_frames) {
    float aspect_ratio = (float)width / height;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int frame = blockIdx.z * blockDim.z + threadIdx.z;
    if (row >= width || col >= height || frame >= num_frames) return;

    float delta = Delta * powf(0.98f, frame);
    const float x0 = xMid - delta * aspect_ratio;
    const float y0 = yMid - delta;
    const float dx = 2.0f * delta * aspect_ratio / width;
    const float dy = 2.0f * delta / height;
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
    } while ((depth > 0) && ((x2 + y2) < 5.0f));
    pic[frame * height * width + row * width + col] = (unsigned char)depth;
}

int main(int argc, char *argv[]) {
    double start, end, elapsed;

    printf("Fractal v2 [parallel]\n");
    int width = atoi(argv[1]);
    int height = atoi(argv[2]);
    int num_frames = atoi(argv[3]);
    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

    unsigned char* pic;
    cudaMallocManaged(&pic, num_frames * height * width * sizeof(unsigned char));

    dim3 threadsPerBlock(16, 16, 4);
    dim3 numBlocks((height + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (width + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (num_frames + threadsPerBlock.z - 1) / threadsPerBlock.z);

    cudaDeviceSynchronize();
    GET_TIME(start);
    fractal<<<numBlocks, threadsPerBlock>>>(pic, width, height, num_frames);
    cudaDeviceSynchronize();
    GET_TIME(end);

    elapsed = end - start;
    printf("Parallel compute time: %.6f s\n", elapsed);

    for (int frame = 0; frame < num_frames; frame++) {
        char name[32];
        sprintf(name, "fractal%d.bmp", frame + 2000);
        writeBMP(width, height, &pic[frame * height * width], name);
    }

    cudaFree(pic);
    return 0;
}
