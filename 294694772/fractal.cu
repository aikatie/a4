/*
 * Computing a movie of zooming into a fractal
 *
 * Original C++ code by Martin Burtscher, Texas State University
 *
 * Reference: E. Ayguade et al., 
 *            "Peachy Parallel Assignments (EduHPC 2018)".
 *                       2018 IEEE/ACM Workshop on Education for High-Performance Computing (EduHPC), pp. 78-85,
 *                                  doi: 10.1109/EduHPC.2018.00012
 *
 *                                  Copyright (c) 2018, Texas State University. All rights reserved.
 *
 *                                  Redistribution and usage in source and binary form, with or without
 *                                  modification, is only permitted for educational use.
 *
 *                                  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *                                  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *                                  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *                                  DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
 *                                  ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *                                  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *                                  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 *                                  ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *                                  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *                                  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 *                                  Author: Martin Burtscher
 *                                  */

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "timer.h"
#include "fractal.h"

static const double Delta = 0.001;
static const double xMid = 0.23701;
static const double yMid = 0.521;

__global__ void compute_fractal(unsigned char *pic, int width, int height, double delta, int frame, double aspect_ratio) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < width && row < height) {
        double x0 = xMid - delta * aspect_ratio + col * (2.0 * delta * aspect_ratio / width);
        double y0 = yMid - delta + row * (2.0 * delta / height);

        double x = 0.0, y = 0.0, x2, y2;
        int depth = 256;
        do {
            x2 = x * x;
            y2 = y * y;
            y = 2 * x * y + y0;
            x = x2 - y2 + x0;
            depth--;
        } while ((depth > 0) && (x2 + y2 < 5.0));

        pic[frame * height * width + row * width + col] = (unsigned char)depth;
    }
}

int main(int argc, char *argv[]) {
    double start, end;

    printf("Fractal v1.6 [CUDA]\n");

    if (argc != 4) {
        fprintf(stderr, "usage: %s width height num_frames\n", argv[0]);
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

    unsigned char *pic = (unsigned char *)malloc(num_frames * height * width * sizeof(unsigned char));
    unsigned char *d_pic;
    cudaMalloc((void **)&d_pic, num_frames * height * width * sizeof(unsigned char));

    GET_TIME(start);

    const double aspect_ratio = (double)width / height;
    double delta = Delta;

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x,
                    (height + threads_per_block.y - 1) / threads_per_block.y);

    for (int frame = 0; frame < num_frames; frame++) {
        compute_fractal<<<num_blocks, threads_per_block>>>(d_pic, width, height, delta, frame, aspect_ratio);
        cudaDeviceSynchronize();
        delta *= 0.98;
    }

    cudaMemcpy(pic, d_pic, num_frames * height * width * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    GET_TIME(end);
    double elapsed = end - start;
    printf("Computed %d frames in %.4f seconds\n", num_frames, elapsed);

    if ((width <= 320) && (num_frames <= 100)) {
        for (int frame = 0; frame < num_frames; frame++) {
            char name[32];
            sprintf(name, "fractal%d.bmp", frame + 1000);
            writeBMP(width, height, &pic[frame * height * width], name);
        }
    }
    cudaFree(d_pic);
    free(pic);

    return 0;
}