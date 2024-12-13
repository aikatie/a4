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

#include "fractal.h"
#include "timer.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static const double Delta = 0.001;
static const double xMid = 0.23701;
static const double yMid = 0.521;

int th_per_blk = 1024;

/* The total number of pixels is calculated by finding the dimensions of each frame (width * height) and multiplying by the number of frames
    The number of blocks is calculated by dividing the total number of pixels by the number of threads per block and rounding up to the nearest integer (+ th_per_blk - 1)

    The frame index is calculated by dividing the cumulative thread index (threads position in the entire sequence of blocks) by the total number of pixels (width * height) in each frame

    delta is originally set to Delta and is multiplied by 0.98 at the end of the iteration of each frame loop, replaced by raising 0.98 to the power of the frame index to achieve equivalent results

    The thread's row is computed by dividing the cumulative thread index by the width (to determine how many rows have been completed) and then taking the modulus of the height (to determine the current row into the frame)

    Dividing by width is necessary first as otherwise, calculations would assume array traversal increments down the rows one by one, not across the columns first and then down the rows

    The thread's column is computed by taking the modulus of the cumulative thread index by the width (to determine the current column into the frame)

    Since columns are traversed first, the modulus is taken by the width to determine the current column, and do not need to account for the height

    Bounds condition checks if the thread is within the bounds of the total number of pixels, and only runs the calculations if the thread is within the bounds (to potentially prevent final block threads from running calculations outside of the bounds)
 */

__global__ void pixelCalc(int width, int height, int frames, double delta, unsigned char *pic) {
    double aspect_ratio = (double)width / height;
    int frameIndex = ((blockIdx.x * blockDim.x) + threadIdx.x) / (width * height); // Calculates assigned frame index for the current thread
    float blockDelta = delta * pow(0.98, frameIndex);                              // Calculates delta for the block's assigned frame
    const double x0 = xMid - blockDelta * aspect_ratio;
    const double y0 = yMid - blockDelta;
    const double dx = 2.0 * blockDelta * aspect_ratio / width;
    const double dy = 2.0 * blockDelta / height;

    int row = (blockIdx.x * blockDim.x + threadIdx.x) / width % height; // Calculates row index for the current thread
    int col = (blockIdx.x * blockDim.x + threadIdx.x) % width;          // Calculates column index for the current thread

    double cy = y0 + row * dy;
    double cx = x0 + col * dx;
    double x = cx;
    double y = cy;
    int depth = 256;
    double x2, y2;

    if (blockDim.x * blockIdx.x + threadIdx.x < frames * width * height) { // Only runs if the thread is within the bounds of the total number of pixels
        do {
            x2 = x * x;
            y2 = y * y;
            y = 2 * x * y + cy;
            x = x2 - y2 + cx;
            depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));
        pic[frameIndex * height * width + row * width + col] = (unsigned char)depth;
    }

    return;
}

int main(int argc, char *argv[]) {
    double start, end;

    printf("Fractal v1.6 [parallel]\n");

    /* read command line arguments */
    if (argc != 4) {
        fprintf(stderr, "usage: %s height width num_frames\n", argv[0]);
        exit(-1);
    }
    int width = atoi(argv[1]);
    if (width < 10) {
        fprintf(stderr, "error: width must be at least 10\n");
        exit(-1);
    }
    int height = atoi(argv[2]);
    if (height < 10) {
        fprintf(stderr, "error: height must be at least 10\n");
        exit(-1);
    }
    int num_frames = atoi(argv[3]);
    if (num_frames < 1) {
        fprintf(stderr, "error: num_frames must be at least 1\n");
        exit(-1);
    }
    printf("Computing %d frames of %d by %d fractal\n", num_frames, width, height);

    /* allocate image array */
    unsigned char *pic;

    /* start time */
    GET_TIME(start);

    cudaMallocManaged(&pic, num_frames * height * width * sizeof(unsigned char));
    int numBlocks = (num_frames * height * width + th_per_blk - 1) / th_per_blk;

    /* compute frames */
    pixelCalc<<<numBlocks, th_per_blk>>>(width, height, num_frames, Delta, pic);
    cudaDeviceSynchronize();

    /* end time */
    GET_TIME(end);
    double elapsed = end - start;
    printf("Serial compute time: %.4f s\n", elapsed);

    /* write frames to BMP files */
    if ((width <= 320) && (num_frames <= 100)) { /* do not write if images large or many */
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic[frame * height * width], name);
        }
    }
    cudaFree(pic);

    return 0;
} /* main */
