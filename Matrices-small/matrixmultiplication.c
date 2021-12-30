#include <stdio.h>
#include <stdlib.h>

int *find_number_of_rows_and_columns(const char *file_name);

void main()
{
    FILE *fp1, *fp2 = NULL;
    int row, col;
    float matval = 0.0;
    int c;

    int rows_in_matrixA, cols_in_matrixA, rows_in_matrixB, cols_in_matrixB, rows_in_matrixC, cols_in_matrixC;

    char *matrixA_filename = "Mat1.txt";
    char *matrixB_filename = "Mat2.txt";

    fp1 = fopen(matrixA_filename, "r");
    fp2 = fopen(matrixB_filename, "r");

    if (fp1 != NULL && fp2 != NULL)
    {
        int *p;
        int *q;

        p = find_number_of_rows_and_columns(matrixA_filename);

        rows_in_matrixA = *(p + 0);
        cols_in_matrixA = *(p + 1);

        // Expected output (from the file `Mat1.txt`) >>> Rows: 8, Columns: 11
        // printf("\nmatA.rows >>> %d", *(p + 0));
        // printf("\nmatA.columns >>> %d", *(p + 1));

        q = find_number_of_rows_and_columns(matrixB_filename);

        rows_in_matrixB = *(q + 0);
        cols_in_matrixB = *(q + 1);

        // output matrix C is the combination of rows from matrix A and columns from matrix B
        rows_in_matrixC = rows_in_matrixA;
        cols_in_matrixC = cols_in_matrixB;

        // Expected output (from the file `Mat2.txt`) >>> Rows: 11, Columns: 8
        // printf("\nmatB.rows >>> %d", *(q + 0));
        // printf("\nmatB.columns >>> %d", *(q + 1));

        printf("\nMatrix A >>> Rows: %d, Columns: %d\n", rows_in_matrixA, cols_in_matrixA);
        printf("Matrix B >>> Rows: %d, Columns: %d\n", rows_in_matrixB, cols_in_matrixB);
        printf("Matrix C >>> Rows: %d, Columns: %d\n", rows_in_matrixA, cols_in_matrixB);

        if (cols_in_matrixA == rows_in_matrixB)
        {
            printf("\nIt bloody well reads and multiplies the matrices rows and columns but does leave out an extra `[]` symbol at the end, nothing more :(\n");

            // elements/values present in each matrix
            int matA_elements = rows_in_matrixA * cols_in_matrixA;
            int matB_elements = rows_in_matrixB * cols_in_matrixB;
            int matC_elements = rows_in_matrixC * cols_in_matrixC;

            // dynamic memory allocation
            float *matrixA_ptr = (float *)malloc(matA_elements * sizeof(float));
            float *matrixB_ptr = (float *)malloc(matB_elements * sizeof(float));
            // float *output_matrix_ptr = (float *)malloc(11 * 11 * sizeof(float));

            if (matrixA_ptr == NULL || matrixB_ptr == NULL)
            {
                printf("\nError! memory not allocated.\n");
                exit(0);
            }

            // Scanning the file and storing the matrix A data in allocated memory
            int counter_matA = 0;
            for (row = 0; row < rows_in_matrixA; row++)
            {
                for (col = 0; col < cols_in_matrixA; col++)
                {
                    fscanf(fp1, "%f,", matrixA_ptr + counter_matA);
                    // printf("R: %d, C: %d, %f  ", row, col, *(matrixA_ptr + col));
                    counter_matA++;
                }
            }

            // Scanning the file and storing the matrix B data in allocated memory
            int counter_matB = 0;
            for (row = 0; row < rows_in_matrixB; row++)
            {
                for (col = 0; col < cols_in_matrixB; col++)
                {
                    fscanf(fp2, "%f,", matrixB_ptr + counter_matB);
                    // printf("R: %d, C: %d, %f  ", row, col, *(matrixA_ptr + col));
                    counter_matB++;
                }
            }

            // printing the elements present in matrix A allocated in the dynamic memory
            int count = 1;
            printf("\nMatrix A elements >>> \n");
            printf("[  ");
            for (int i = 0; i < matA_elements; i++)
            {
                printf("%f  ", *(matrixA_ptr + i));
                count++;

                if (count == cols_in_matrixA + 1)
                {
                    printf("]\n[  ");
                    count = 1;
                }
            }
            printf("\n\n");

            // printing the elements present in matrix B allocated in the dynamic memory
            count = 1;
            printf("\nMatrix B elements >>> \n");
            printf("[  ");
            for (int i = 0; i < matB_elements; i++)
            {
                printf("%f  ", *(matrixB_ptr + i));
                count++;

                if (count == cols_in_matrixB + 1)
                {
                    printf("]\n[  ");
                    count = 1;
                }
            }
            printf("\n\n");

            // algorithm created to multiply Matrix A and Matrix B resulting in an output as Matrix C
            // float output_matrix[10][10];

            float output_matrix_array[matC_elements];

            for (int i = 0; i < rows_in_matrixA; i++)
            {
                for (int j = 0; j < cols_in_matrixB; j++)
                {
                    float sum = 0.0;
                    for (int k = 0; k < rows_in_matrixB; k++)
                        sum = sum + *(matrixA_ptr + (i * cols_in_matrixA + k)) * *(matrixB_ptr + (k * cols_in_matrixB + j));
                    // (output_matrix_ptr + (i * 11 + j)) = sum;
                    output_matrix_array[i * cols_in_matrixC + j] = sum; // 3 is the number of columns in matrix C
                }
            }

            //////////////// TEST PURPOSES ONLY /////////////////////
            // for (int i = 0; i <= matC_elements; i++)
            // {
            //     printf("\noutput_matrix[%d] = %f", i, output_matrix_array[i]);
            // }
            // printf("\n\n");
            //////////////// TEST PURPOSES ONLY /////////////////////

            // printing the output matrix C stored inside an array
            count = 1;
            printf("\nOutput matrix C elements >>> \n");
            printf("[  ");
            for (int i = 0; i < matC_elements; i++)
            {
                printf("%f  ", output_matrix_array[i]);
                count++;

                if (count == cols_in_matrixC + 1)
                {
                    printf("]\n[  ");
                    count = 1;
                }
            }
            printf("\n");

            // deallocating the memory
            free(matrixA_ptr);
            free(matrixB_ptr);
            // free(output_matrix_ptr);
        }
        else
        {
            printf("\nOops! the column of matrix A is not equal to the row of matrix B, thus matrices cannot be multiplied.\n");
        }

        fclose(fp1);
        fclose(fp2);
    }
    else
    {
        printf("\nNo such file found!\n");
    }
}

// function to find the number of rows and columns of each matrix
int *find_number_of_rows_and_columns(const char *file_name)
{
    FILE *fp = fopen(file_name, "r");
    int newRows = 1;
    int newCols = 1;
    char ch;

    static int rows_cols[10];

    while (!feof(fp))
    {
        ch = fgetc(fp);

        if (ch == '\n')
        {
            newRows++;
            // rows_cols[0] = newCols;
            newCols = 1;
        }
        else if (ch == ',')
        {
            newCols++;
        }
    }
    rows_cols[0] = newRows;
    rows_cols[1] = newCols;

    // printf("\nRows: %d, Cols: %d\n", rows_cols[0], rows_cols[1]);

    return rows_cols;
}
