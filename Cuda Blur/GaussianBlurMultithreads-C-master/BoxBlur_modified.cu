#include <stdio.h>
#include <stdlib.h>
#include "lodepng.h"

unsigned char *output_img;




__global__ void BoxBlur(unsigned char *gpu_image_output, unsigned char *gpu_image_input, int w, int h)
{
	int counter = 0;

	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int i = blockIdx.x;
	int j = threadIdx.x;
	int n = 0;


	// for (i = startLimit; i < endLimit; i += 4)
	// {
		//left
		if ((i % (w * 4)) == 0)
		{

			//top left corner
			if (i == 0)
			{
				n = 4;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (gpu_image_input[i + j] + gpu_image_input[i + 4 + j] + gpu_image_input[4 * w + j] + gpu_image_input[4 * w + 4 + j]) / n;
				}
				*(output_img + i + 3) = gpu_image_input[i + 3];
			}

			//bottom left corner
			else if (i == (w * 4 * (h - 1)))
			{
				n = 4;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (gpu_image_input[i + j] + gpu_image_input[i + 4 + j] + gpu_image_input[i - (4 * w) + j] + gpu_image_input[i - (4 * w) + 4 + j]) / n;
				}
				*(output_img + i + 3) = gpu_image_input[i + 3];
			}

			//Pure left
			else
			{
				n = 6;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (gpu_image_input[i + j] + gpu_image_input[i + 4 + j] + gpu_image_input[(4 * w) + i + j] + gpu_image_input[(4 * w) + i + 4 + j] + gpu_image_input[i - (4 * w) + j] + gpu_image_input[i - (4 * w) + 4 + j]) / n;
				}
				*(output_img + i + 3) = gpu_image_input[i + 3];
			}
		}

		//right
		else if ((i % (w * 4)) == (w * 4) - 4)
		{

			//top right corner
			if (i == ((w * 4) - 4))
			{
				n = 4;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (gpu_image_input[i + j] + gpu_image_input[i - 4 + j] + gpu_image_input[(4 * w) + i + j] + gpu_image_input[(4 * w) + i - 4 + j]) / n;
				}
				*(output_img + i + 3) = gpu_image_input[i + 3];
			}

			//bottom right corner
			else if (i == ((w * h * 4) - 4))
			{
				n = 4;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (gpu_image_input[i + j] + gpu_image_input[i - 4 + j] + gpu_image_input[i - (4 * w) + j] + gpu_image_input[i - (4 * w) - 4 + j]) / n;
				}
				*(output_img + i + 3) = gpu_image_input[i + 3];
			}

			//Pure right
			else
			{
				n = 6;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (gpu_image_input[i + j] + gpu_image_input[i - 4 + j] + gpu_image_input[(4 * w) + i + j] + gpu_image_input[(4 * w) + i - 4 + j] + gpu_image_input[i - (4 * w) + j] + gpu_image_input[i - (4 * w) - 4] + j) / n;
				}
				*(output_img + i + 3) = gpu_image_input[i + 3];
			}
		}

		//top
		else if (i > 0 && i < ((w * 4) - 4))
		{
			n = 6;
			for (j = 0; j < 3; j++)
			{
				*(output_img + i + j) = (gpu_image_input[i + j] + gpu_image_input[i - 4 + j] + gpu_image_input[i + 4 + j] + gpu_image_input[(4 * w) + i + j] + gpu_image_input[(4 * w) + i + 4 + j] + gpu_image_input[(4 * w) + i - 4 + j]) / n;
			}
			*(output_img + i + 3) = gpu_image_input[i + 3];
		}

		//bottom
		else if (i > (w * 4 * (h - 1)) && i < ((w * h * 4) - 4))
		{
			n = 6;
			for (j = 0; j < 3; j++)
			{
				*(output_img + i + j) = (gpu_image_input[i + j] + gpu_image_input[i - 4 + j] + gpu_image_input[i + 4 + j] + gpu_image_input[i - (4 * w) + j] + gpu_image_input[i - (4 * w) + 4 + j] + gpu_image_input[i - (4 * w) - 4 + j]) / n;
			}
			*(output_img + i + 3) = gpu_image_input[i + 3];
		}

		//middle
		else
		{
			n = 9;
			for (j = 0; j < 3; j++)
			{
				*(output_img + i + j) = (gpu_image_input[i + j] + gpu_image_input[i + 4 + j] + gpu_image_input[i - 4 + j] + gpu_image_input[(4 * w) + i + j] + gpu_image_input[(4 * w) + i + 4 + j] + gpu_image_input[(4 * w) + i - 4 + j] + gpu_image_input[i - (4 * w) + j] + gpu_image_input[i - (4 * w) + 4 + j] + gpu_image_input[i - (4 * w) - 4 + j]) / n;
			}
			*(output_img + i + 3) = gpu_image_input[i + 3];
		}
	// }
}

void main()
{
	int i, j;
	unsigned error;
	unsigned char *image;
	unsigned w, h;
	char *filename = "hck.png";
	char *output_filename = "hck_blurred.png";

	error = lodepng_decode32_file(&gpu_image_input, &w, &h, filename);
	if (error)
	{
		printf("Error decoding image %u : %s\n", error, lodepng_error_text(error));
	}
	else
	{
		printf("Height: %d pixels,\tWidth: %d pixels\n\n", h, w);
	}

	output_img = (unsigned char *)malloc(w * h * 4 * sizeof(char));

	int ARRAY_SIZE = w * h * 4;
	int ARRAY_BYTES = ARRAY_SIZE * sizeof(unsigned char);

	unsigned char host_image_input[ARRAY_SIZE * 4];
	unsigned char host_image_output[ARRAY_SIZE * 4];

	for (int i = 0; i < ARRAY_SIZE; i++) {
		host_image_input[i] = image[i];
	}

	// device/GPU memory pointers declaration
	unsigned char *d_in;
	unsigned char *d_out;

	// device memory allocation
	cudaMalloc((void**) &d_in, ARRAY_BYTES);
	cudaMalloc((void**) &d_out, ARRAY_BYTES);

	cudaMemcpy(d_in, host_image_input, ARRAY_BYTES, cudaMemcpyHostToDevice);

	// launch the kernel
	BoxBlur<<<h, w>>>(d_out, d_in, w, h);

	// copy back the result array to the CPU
	cudaMemcpy(host_image_output, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost);

	unsigned char *png;
	size_t pngsize;
	error = lodepng_encode32(&png, &pngsize, output_img, w, h);

	if (!error)
	{
		lodepng_save_file(png, pngsize, filename);
		printf("Encoding successful!\nThe image is saved in the same directory :).");
	}

	//freeing the memory allocated by the output_img array.
	free(output_img);
}
