// // Pointers can be easily used to create a 2D array in C using malloc. The idea is to first create a one dimensional array of pointers, and then, for each array entry, // create another one dimensional array. Here's a sample code:
// double **theArray;
// theArray = (double **)malloc(arraySizeX * sizeof(double *));
// for (int i = 0; i < arraySizeX; i++)
//     theArray[i] = (double *)malloc(arraySizeY * sizeof(double));

// // What I usually do is create a function called Make2DDoubleArray that returns a (double**) and then use it in my code to declare 2D arrays here and there
// double **Make2DDoubleArray(int arraySizeX, int arraySizeY)
// {
//     double **theArray;
//     theArray = (double **)malloc(arraySizeX * sizeof(double *));
//     for (int i = 0; i < arraySizeX; i++)
//         theArray[i] = (double *)malloc(arraySizeY * sizeof(double));
//     return theArray;
// }

// // Then, inside the code, i would use something like
// double **myArray = Make2DDoubleArray(nx, ny);

// // Of course, do not forget to remove your arrays from memory once you're done using them. To do this
// for (i = 0; i < nx; i++)
// {
//     free(myArray[i]);
// }
// free(myArray);