#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include "lodepng.h"

unsigned char *output_img;
unsigned char *Image;
unsigned w, h;

//creating a variable for multithreading iterations
struct variables
{
	unsigned int start;
	unsigned int end;
};

//creating a multitheaded function for gaussian blur
void *GaussianBlur(void *vars)
{
	int i, j;
	int n = 0;
	struct variables *val = (struct variables *)vars;
	unsigned int startLimit = val->start;
	unsigned int endLimit = val->end;

	//Identifying the pixels by iterating through the image and saving the changes on another array that is dynamically allocated.

	for (i = startLimit; i < endLimit; i += 4)
	{
		//left
		if ((i % (w * 4)) == 0)
		{

			//top left corner
			if (i == 0)
			{
				n = 4;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (Image[i + j] + Image[i + 4 + j] + Image[4 * w + j] + Image[4 * w + 4 + j]) / n;
				}
				*(output_img + i + 3) = Image[i + 3];
			}

			//bottom left corner
			else if (i == (w * 4 * (h - 1)))
			{
				n = 4;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (Image[i + j] + Image[i + 4 + j] + Image[i - (4 * w) + j] + Image[i - (4 * w) + 4 + j]) / n;
				}
				*(output_img + i + 3) = Image[i + 3];
			}

			//Pure left
			else
			{
				n = 6;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (Image[i + j] + Image[i + 4 + j] + Image[(4 * w) + i + j] + Image[(4 * w) + i + 4 + j] + Image[i - (4 * w) + j] + Image[i - (4 * w) + 4 + j]) / n;
				}
				*(output_img + i + 3) = Image[i + 3];
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
					*(output_img + i + j) = (Image[i + j] + Image[i - 4 + j] + Image[(4 * w) + i + j] + Image[(4 * w) + i - 4 + j]) / n;
				}
				*(output_img + i + 3) = Image[i + 3];
			}

			//bottom right corner
			else if (i == ((w * h * 4) - 4))
			{
				n = 4;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (Image[i + j] + Image[i - 4 + j] + Image[i - (4 * w) + j] + Image[i - (4 * w) - 4 + j]) / n;
				}
				*(output_img + i + 3) = Image[i + 3];
			}

			//Pure right
			else
			{
				n = 6;
				for (j = 0; j < 3; j++)
				{
					*(output_img + i + j) = (Image[i + j] + Image[i - 4 + j] + Image[(4 * w) + i + j] + Image[(4 * w) + i - 4 + j] + Image[i - (4 * w) + j] + Image[i - (4 * w) - 4] + j) / n;
				}
				*(output_img + i + 3) = Image[i + 3];
			}
		}

		//top
		else if (i > 0 && i < ((w * 4) - 4))
		{
			n = 6;
			for (j = 0; j < 3; j++)
			{
				*(output_img + i + j) = (Image[i + j] + Image[i - 4 + j] + Image[i + 4 + j] + Image[(4 * w) + i + j] + Image[(4 * w) + i + 4 + j] + Image[(4 * w) + i - 4 + j]) / n;
			}
			*(output_img + i + 3) = Image[i + 3];
		}

		//bottom
		else if (i > (w * 4 * (h - 1)) && i < ((w * h * 4) - 4))
		{
			n = 6;
			for (j = 0; j < 3; j++)
			{
				*(output_img + i + j) = (Image[i + j] + Image[i - 4 + j] + Image[i + 4 + j] + Image[i - (4 * w) + j] + Image[i - (4 * w) + 4 + j] + Image[i - (4 * w) - 4 + j]) / n;
			}
			*(output_img + i + 3) = Image[i + 3];
		}

		//middle
		else
		{
			n = 9;
			for (j = 0; j < 3; j++)
			{
				*(output_img + i + j) = (Image[i + j] + Image[i + 4 + j] + Image[i - 4 + j] + Image[(4 * w) + i + j] + Image[(4 * w) + i + 4 + j] + Image[(4 * w) + i - 4 + j] + Image[i - (4 * w) + j] + Image[i - (4 * w) + 4 + j] + Image[i - (4 * w) - 4 + j]) / n;
			}
			*(output_img + i + 3) = Image[i + 3];
		}
	}
}

void main()
{
	int i, j;
	unsigned error;
	char filename[20];
	char output_filename[20];

	printf("Enter input file name: (Example: picture1.png)\nNote: Specify the directory if the picture is in different folder.\n");
	scanf("%s", filename);

	error = lodepng_decode32_file(&Image, &w, &h, filename);
	if (error)
	{
		printf("Error decoding image %u : %s\n", error, lodepng_error_text(error));
	}
	else
	{
		printf("Height of image is: %d pixels and width of image is: %d pixels.\n\n", h, w);
	}

	printf("Enter output file name: (Example: picture2.png)\n");
	scanf("%s", filename);

	output_img = (char *)malloc(w * h * 4 * sizeof(char));

	//Slicing for equal division of tasks among the threads
	long iterations = w * h * 4;
	int threads;

	printf("Enter the number of threads: ");
	scanf("%d", &threads);

	int sliceList[threads];
	int remainder = iterations % threads;

	for (i = 0; i < threads; i++)
	{
		sliceList[i] = iterations / threads;
	}

	for (j = 0; j < remainder; j++)
	{
		sliceList[j] += 1;
	}

	int startList[threads];
	int endList[threads];

	int l;
	for (l = 0; l < threads; l++)
	{
		if (l == 0)
			startList[l] = 0;
		else
			startList[l] = endList[l - 1] + 1;

		endList[l] = startList[l] + sliceList[l] - 1;
	}

	struct variables arr1[threads];

	int k;
	for (k = 0; k < threads; k++)
	{
		arr1[k].start = startList[k];
		arr1[k].end = endList[k];
	}

	pthread_t threadId[threads];

	int m;
	for (m = 0; m < threads; m++)
	{
		pthread_create(&threadId[m], NULL, GaussianBlur, &arr1[m]);
	}

	int nn;
	for (nn = 0; nn < threads; nn++)
	{
		pthread_join(threadId[nn], NULL);
	}

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
