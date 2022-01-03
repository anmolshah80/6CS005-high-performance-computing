#include <stdio.h>
#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

// Compile with:  nvcc cudablur.cu lodepng.cpp -o cudablur
// Execute with: ./cudablur

__global__ void box_blur(unsigned char * device_image_output, unsigned char * device_image_input, unsigned int width, unsigned int height)
{
	int r = 0;
	int g = 0;
	int b = 0;
	int a = 0;
	int x, y;
	int count = 0;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int pixel = idx * 4;

	for (x = (pixel - 4); x <= (pixel + 4); x += 4) 
	{
		if ((x > 0) && x < (height * width * 4) && ((x - 4) / (4 * width) == pixel / (4 * width))) 
		{
			for (y = (x - (4 * width)); y <=  (x + (4 * width)); y += (4 * width)) 
			{
				if (y > 0 && y < (height * width * 4)) 
				{
					r += device_image_input[y];
					g += device_image_input[1 + y];
					b += device_image_input[2 + y]; 
					count++;
				}
			}
		}
	}
	
	a = device_image_input[3 + pixel];

	device_image_output[pixel] = r / count;
	device_image_output[1 + pixel] = g / count;
	device_image_output[2 + pixel] = b / count;
	device_image_output[3 + pixel] = a;
}

int main(int argc, char **argv)
{
	unsigned int error;
	unsigned int encError;
	unsigned char* image;
	unsigned int width;
	unsigned int height;
	const char* filename = "hck.png";
	const char* newFileName = "blurredHCK_2.png";

	error = lodepng_decode32_file(&image, &width, &height, filename);
	if (error) {
		printf("Error %u: %s\n", error, lodepng_error_text(error));
	}

	const int ARRAY_SIZE = width * height * 4;
	const int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned char);

	unsigned char host_imageInput[ARRAY_SIZE * 4];
	unsigned char host_imageOutput[ARRAY_SIZE * 4];

	for (int i = 0; i < ARRAY_SIZE; i++) {
		host_imageInput[i] = image[i];
	}

	// declaring device memory pointers
	unsigned char * d_in;
	unsigned char * d_out;

	// allocating device memory
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	cudaMemcpy(d_in, host_imageInput, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launching the kernel function
	box_blur<<<height, width>>>(d_out, d_in, width, height);

	// copy the computed result array back to the host array
	cudaMemcpy(host_imageOutput, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);
	
	encError = lodepng_encode32_file(newFileName, host_imageOutput, width, height);
	if (encError) {
		printf("error %u: %s\n", error, lodepng_error_text(encError));
	}

	// deallocating the device memory
	cudaFree(d_in);
	cudaFree(d_out);

	return 0;
}
