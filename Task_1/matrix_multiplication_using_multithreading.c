#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// Compile with: gcc matrix_multiplication_using_multithreading.c -pthread -o matrix_multiplication_using_multithreading
// Run with: matrix_multiplication_using_multithreading <number_of_threads>
//          where number of threads > 0 and < 1000

int threadCount;

// structure to store the individual martix elements
typedef struct
{
    double x;
} unit;

// structure to store the individual matrix rows, columns
// and the reference to the matrix elements
typedef struct
{
    int rows;
    int cols;
    unit *x;
} matrix;

matrix A, B, C, _C, target;

pthread_mutex_t mutex;

// function prototypes
void inputArguments(char *program_name);
void getArguments(int argc, char *argv[]);
void max_threads_allowed(int rows_matA, int cols_matA, int rows_matB, int cols_matB);
int *find_number_of_rows_and_columns(const char *file_name);
matrix create_matrix(int rows, int cols);
void display_matrix(matrix displayable_matrix);
double calculate_one_matrix_unit(int first, int second);
void *multiply_matrices(void *args);

// the main function that invokes itself at runtime, obviously!
void main(int argc, char *argv[])
{
    getArguments(argc, argv);

    FILE *fp1, *fp2, *fp3 = NULL;
    int row, col;

    int rows_in_matrixA, cols_in_matrixA, rows_in_matrixB, cols_in_matrixB, rows_in_matrixC, cols_in_matrixC;

    char *matrixA_filename = "Mat1.txt";
    char *matrixB_filename = "Mat2.txt";
    char *output_matrix_filename = "matrixresults2050423.txt";

    fp1 = fopen(matrixA_filename, "r");
    fp2 = fopen(matrixB_filename, "r");
    fp3 = fopen(output_matrix_filename, "w");

    if (fp1 != NULL && fp2 != NULL && fp3 != NULL)
    {
        int *p;
        int *q;

        p = find_number_of_rows_and_columns(matrixA_filename);

        rows_in_matrixA = *(p + 0);
        cols_in_matrixA = *(p + 1);

        //////////////////// TEST PURPOSES ONLY //////////////////////////
        // Expected output (from the file `Mat1.txt`) >>> Rows: 8, Columns: 11
        // printf("\nmatA.rows >>> %d", *(p + 0));
        // printf("\nmatA.columns >>> %d", *(p + 1));
        //////////////////// TEST PURPOSES ONLY //////////////////////////

        q = find_number_of_rows_and_columns(matrixB_filename);

        rows_in_matrixB = *(q + 0);
        cols_in_matrixB = *(q + 1);

        max_threads_allowed(rows_in_matrixA, cols_in_matrixA, rows_in_matrixB, cols_in_matrixB);

        // output matrix C is the combination of rows from matrix A and columns from matrix B
        rows_in_matrixC = rows_in_matrixA;
        cols_in_matrixC = cols_in_matrixB;

        //////////////////// TEST PURPOSES ONLY //////////////////////////
        // Expected output (from the file `Mat2.txt`) >>> Rows: 11, Columns: 8
        // printf("\nmatB.rows >>> %d", *(q + 0));
        // printf("\nmatB.columns >>> %d", *(q + 1));
        //////////////////// TEST PURPOSES ONLY //////////////////////////

        printf("\nMatrix A >>> Rows: %d, Columns: %d\n", rows_in_matrixA, cols_in_matrixA);
        printf("Matrix B >>> Rows: %d, Columns: %d\n", rows_in_matrixB, cols_in_matrixB);
        printf("Matrix C >>> Rows: %d, Columns: %d\n\n", rows_in_matrixA, cols_in_matrixB);

        if (cols_in_matrixA == rows_in_matrixB)
        {
            matrix target_matA, target_matB;
            target_matA.rows = rows_in_matrixA;
            target_matA.cols = cols_in_matrixA;

            target_matB.rows = rows_in_matrixB;
            target_matB.cols = cols_in_matrixB;

            // dynamic memory allocation
            target_matA.x = (unit *)malloc(rows_in_matrixA * cols_in_matrixA * sizeof(unit));
            target_matB.x = (unit *)malloc(rows_in_matrixB * cols_in_matrixB * sizeof(unit));

            if (target_matA.x == NULL || target_matB.x == NULL)
            {
                printf("\nError! memory not allocated.\n");
                exit(1);
            }

            // Scanning the file and storing the matrix A data in allocated memory
            for (row = 0; row < rows_in_matrixA; row++)
            {
                for (col = 0; col < cols_in_matrixA; col++)
                {
                    fscanf(fp1, "%lf,", &(target_matA.x + row * target_matA.cols + col)->x);
                }
            }

            // printing the matrix A elements
            printf("\nMatrix A elements >>> \n");
            A = target_matA;
            display_matrix(A);

            // Scanning the file and storing the matrix B data in allocated memory
            for (row = 0; row < rows_in_matrixB; row++)
            {
                for (col = 0; col < cols_in_matrixB; col++)
                {
                    fscanf(fp2, "%lf,", &(target_matB.x + row * target_matB.cols + col)->x);
                }
            }

            // printing the matrix B elements
            printf("Matrix B elements >>> \n");
            B = target_matB;
            display_matrix(B);

            int i;
            C = create_matrix(A.rows, B.cols);
            for (i = 0; i < C.cols * C.rows; i++)
            {
                (C.x + i)->x = 0.0;
            }

            _C = create_matrix(A.rows, B.cols);
            for (i = 0; i < _C.cols * _C.rows; i++)
            {
                (_C.x + i)->x = 0.0;
            }

            pthread_t thread_id[threadCount];

            printf("Creating threads and computing the matrix multiplication...\n\n");

            pthread_mutex_init(&mutex, NULL);

            for (int m = 0; m < threadCount; m++)
            {
                pthread_create(&thread_id[m], NULL, multiply_matrices, NULL);
            }

            for (int n = 0; n < threadCount; n++)
            {
                pthread_join(thread_id[n], NULL);
            }

            // writing the output matrix C into the file `matrixresults2050423.txt`
            for (i = 0; i < rows_in_matrixC; i++)
            {
                for (int j = 0; j < cols_in_matrixC; j++)
                {
                    if (j == cols_in_matrixC - 1)
                        fprintf(fp3, "%lf", (C.x + i * cols_in_matrixC + j)->x);
                    else
                        fprintf(fp3, "%lf,", (C.x + i * cols_in_matrixC + j)->x);
                }
                fprintf(fp3, "\n");
            }

            printf("\nOutput matrix C elements >>> \n");
            display_matrix(C);

            printf("Output matrix elements written in the file `matrixresults2050423.txt`\n\n");

            // deallocating the memory
            free(target_matA.x);
            free(target_matB.x);
            free(target.x);
        }
        else
        {
            printf("\nOops! the column of matrix A is not equal to the row of matrix B, thus matrices cannot be multiplied.\n");
        }

        fclose(fp1);
        fclose(fp2);
        fclose(fp3);
    }
    else
    {
        printf("\nNo such file found!\n");
    }
}

// function to display a message explaining what and how arguments should be passed
void inputArguments(char *program_name)
{
    fprintf(stderr, "arguments should be in the order as specified:   %s   <number of threads>\n", program_name);
    fprintf(stderr, "where number of threads should be > 0 and < 1000\n");
    exit(1);
}

// function to get the command line arguments
void getArguments(int argc, char *argv[])
{
    if (argc != 2)
    {
        inputArguments(argv[0]);
    }

    threadCount = strtol(argv[1], NULL, 10);

    if (threadCount <= 0 || threadCount >= 1000)
    {
        inputArguments(argv[0]);
    }
}

// function to handle and limit the number of input threads to the dimensions of the matrices
void max_threads_allowed(int rows_matA, int cols_matA, int rows_matB, int cols_matB)
{
    if (threadCount > rows_matA * cols_matA || threadCount > rows_matB * cols_matB)
    {
        // limiting the MAX_THREADS_ALLOWED to the maximum dimension of matrices
        if (rows_matA * cols_matA > rows_matB * cols_matB)
            threadCount = rows_matA * cols_matA;
        else
            threadCount = rows_matB * cols_matB;
        printf("\nMAX_THREADS_ALLOWED: %d\n", threadCount);
        // printf("since input number of threads should not be greater than the number of rows or columns to be processed\n");
    }
}

// function to find the number of rows and columns of each matrix from the files
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

matrix create_matrix(int rows, int cols)
{
    // matrix target;
    int i, j;
    double temp_data;

    target.rows = rows;
    target.cols = cols;
    target.x = (unit *)malloc(rows * cols * sizeof(unit));
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
        {
            temp_data = 0.0;
            (target.x + i * target.cols + j)->x = temp_data;
        }
    return target;
}

void display_matrix(matrix displayable_matrix)
{
    int rows = displayable_matrix.rows;
    int cols = displayable_matrix.cols;
    int i, j;

    for (i = 0; i < rows; i++)
    {
        printf("[  ");
        for (j = 0; j < cols; j++)
            printf("%f  ", (displayable_matrix.x + i * cols + j)->x);
        printf("]\n");
    }
    printf("\n\n");
}

// function to return a multiplied matrix unit
double calculate_one_matrix_unit(int first, int second)
{
    int i;
    double res = 0.0;

    // to test the thread running status
    // pthread_mutex_lock(&mutex);
    // printf("\n%d,%d is working\n", first, second);
    // pthread_mutex_unlock(&mutex);

    for (i = 0; i < A.cols; i++)
    {
        res += (A.x + first * A.cols + i)->x * (B.x + i * B.cols + second)->x;
        // testing purposes only
        // printf("\nA.x: %d + first: %d * A.cols: %d + i: %d", *(A.x), first, A.cols, i);
        // printf("\nB.x: %d + i: %d * B.cols: %d + second: %d", *(B.x), i, B.cols, second);
        // printf("\n(A.x + first * A.cols + i)->x: %f * (B.x + i * B.cols + second)->x: %f", (A.x + first * A.cols + i)->x, (B.x + i * B.cols + second)->x);
    }

    return res;
}

// thread function to process the number of rows assigned to each thread
void *multiply_matrices(void *param)
{
    while (1)
    {
        int firstNum;
        int secondNum;
        int i, j, flag = 0, close = 0;
        double res;

        pthread_mutex_lock(&mutex);
        for (i = 0; i < _C.rows; i++)
        {
            for (j = 0; j < _C.cols; j++)
            {
                if ((_C.x + i * _C.cols + j)->x == 0.0)
                {
                    firstNum = i;
                    secondNum = j;
                    (_C.x + i * _C.cols + j)->x = 1.0;
                    close = 1;
                    break;
                }
            }
            if (close == 1)
                break;
            else if (i == _C.rows - 1)
                flag = 1;
        }
        pthread_mutex_unlock(&mutex);

        if (flag == 1)
            pthread_exit(NULL);
        res = calculate_one_matrix_unit(firstNum, secondNum);
        (C.x + firstNum * C.cols + secondNum)->x = res;
    }
    pthread_exit(NULL);
}