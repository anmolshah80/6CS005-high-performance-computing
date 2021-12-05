# 6CS005 High Performance Computing

- Matrix multiplication using multithreading

  - Reading data from the file [done]
  - Using dynamic memory (malloc) for matrix A and matrix B [done]
  - Creating an algorithm to multiply matrices correctly [done]
  - Using multithreading with equal computations
  - Storing the correct output matrices in the correct format in a file

- Password Cracking using multithreading (POSIX threads)

  - Cracks a password using multithreading and dynamic slicing based on thread count [done]
  - Program finishes appropriately when password has been found

- Password Cracking using CUDA

  - Generate encrypted password in the kernel function (using CudaCrypt function) to be compared to original encrypted password
  - Allocating the correct amount of memory on the GPU based on input data. Memory is freed once used
  - Program works with multiple blocks and threads – the number of blocks and threads will depend on your kernel function. You will not be penalised if your program only works with a set number of blocks and threads however, your program must use more than one block (axis is up to you) and more than one thread (axis is up to you)
  - Decrypted password sent back to the CPU and printed

- Box Blur using CUDA
  - Reading in an image file into a single or 2D array
  - Allocating the correct amount of memory on the GPU based on input data. Memory is freed once used
  - Applying Box filter on image in the kernel function
  - Return blurred image data from the GPU to the CPU
  - Outputting the correct image with Box Blur applied as a file
