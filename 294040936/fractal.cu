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

THIS SOFTWARE IS PROVIDED "AS IS" WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
*/

#include <stdlib.h>
#include <stdio.h>
#include "timer.h"
#include "fractal.h"

// Constants for fractal computation
static const double BASE_DELTA = 0.001;  // Initial zoom step
static const double CENTER_X = 0.23701; // Fractal center X
static const double CENTER_Y = 0.521;   // Fractal center Y
static const int MAX_ITER = 256;        // Maximum iteration limit

__device__ double computeDelta(int frameIndex) {
    // Compute zoom delta for a specific frame
    return BASE_DELTA * pow(0.98, frameIndex);
}

__global__ void generateFractalImage(unsigned char *imageBuffer, int imgWidth, int imgHeight, int totalFrames) {
    int pixelIndex = blockIdx.x * blockDim.x + threadIdx.x;

    // Avoid out-of-bounds memory access
    if (pixelIndex >= imgWidth * imgHeight * totalFrames) return;

    // Derive frame, row, and column for the current pixel
    int frameIndex = pixelIndex / (imgWidth * imgHeight);
    int row = (pixelIndex / imgWidth) % imgHeight;
    int col = pixelIndex % imgWidth;

    // Calculate fractal parameters
    double aspectRatio = (double)imgWidth / imgHeight;
    double delta = computeDelta(frameIndex);

    double startX = CENTER_X - delta * aspectRatio;
    double startY = CENTER_Y - delta;
    double stepX = 2.0 * delta * aspectRatio / imgWidth;
    double stepY = 2.0 * delta / imgHeight;

    // Compute pixel coordinates in the complex plane
    double pixelX = startX + col * stepX;
    double pixelY = startY + row * stepY;

    // Mandelbrot iteration
    double x = pixelX, y = pixelY;
    double xSquared = 0.0, ySquared = 0.0;
    int iterations = MAX_ITER;

    while (iterations > 0 && (xSquared + ySquared) < 4.0) {
        y = 2.0 * x * y + pixelY;
        x = xSquared - ySquared + pixelX;
        xSquared = x * x;
        ySquared = y * y;
        iterations--;
    }

    // Store the iteration count as a grayscale value
    imageBuffer[pixelIndex] = (unsigned char)iterations;
}

int main(int argc, char *argv[]) {
    printf("Fractal Renderer v2.0 [CUDA Parallelized]\n");

    // Validate input arguments
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <width> <height> <total_frames>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int imgWidth = atoi(argv[1]);
    int imgHeight = atoi(argv[2]);
    int totalFrames = atoi(argv[3]);

    if (imgWidth < 10 || imgHeight < 10 || totalFrames < 1) {
        fprintf(stderr, "Error: Width and height must be at least 10, and frames must be at least 1.\n");
        return EXIT_FAILURE;
    }

    printf("Rendering %d frames of size %d x %d\n", totalFrames, imgWidth, imgHeight);

    // Allocate memory for the fractal image
    unsigned char *imageData;
    cudaMallocManaged(&imageData, totalFrames * imgWidth * imgHeight * sizeof(unsigned char));

    // Start the timer
    double startTime, endTime;
    GET_TIME(startTime);

    // Launch the kernel to generate the fractal
    int totalPixels = imgWidth * imgHeight * totalFrames;
    int threadsPerBlock = 256;
    int blocksRequired = (totalPixels + threadsPerBlock - 1) / threadsPerBlock;

    generateFractalImage<<<blocksRequired, threadsPerBlock>>>(imageData, imgWidth, imgHeight, totalFrames);
    cudaDeviceSynchronize(); // Wait for kernel execution to complete

    // Stop the timer
    GET_TIME(endTime);
    printf("Fractal computation completed in: %.4f seconds\n", endTime - startTime);

    // Save images as BMP files if the size is manageable
    if (imgWidth <= 320 && totalFrames <= 100) {
        for (int frame = 0; frame < totalFrames; frame++) {
            char filename[32];
            sprintf(filename, "fractal_frame_%d.bmp", frame + 1000);
            writeBMP(imgWidth, imgHeight, &imageData[frame * imgWidth * imgHeight], filename);
        }
    }

    // Free allocated memory
    cudaFree(imageData);

    return EXIT_SUCCESS;
}
