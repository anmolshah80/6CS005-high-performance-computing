#include <stdio.h>
#include <stdlib.h>
// #include <string.h>
#include <pthread.h>
// #include <semaphore.h>
#include <unistd.h>
#include <time.h>

void inputArguments(char *program_name);
void getArguments(int argc, char *argv[]);
int *find_number_of_rows_and_columns(const char *file_name);
void *multiply_matrices(void *args);

int max_threads_allowed; // to limit the number of input threads to the dimensions of the matrices

int threadCount;

typedef struct
{
    float x;
} unit;

typedef struct
{
    int rows;
    int cols;
    unit *x;
} matrix;

matrix A, B, C, _C;
pthread_mutex_t mutex;

// to get the start and end value of the iterative loop for each thread
struct threadInfo
{
    int start;
    int end;
};

int thread_counter = 0;
float matrixA_array[100];

// structure to pass multiple arguments inside pthread_create() while calling the thread function
struct args_struct
{
    int matC_elements;
    int rows_matA;
    int cols_matA;
    int rows_matB;
    int cols_matB;
    float **matrixA_ptr;
    float **matrixB_ptr;
    // float **output_matrix_ptr;
    // int thread_count;
    // struct threadInfo threadDetails[100];

    int start;
    int end;
} * __args;

float output_matrix_array[1000];

// sem_t sem;

matrix create_matrix(int rows, int cols)
{
    matrix target;
    int i, j;
    int data;

    target.rows = rows;
    target.cols = cols;
    target.x = (unit *)malloc(rows * cols * sizeof(unit));
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++)
        {
            data = rand() % 100;
            (target.x + i * target.cols + j)->x = (float)data;
        }
    return target;
}

void show_matrix(matrix shows)
{
    int rows = shows.rows;
    int cols = shows.cols;
    int i, j;

    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
            printf("%f\t", (shows.x + i * cols + j)->x);
        printf("\n");
    }
}

int CalcuOneUnit(int first, int second)
{
    int i, res = 0;

    // Here for testing thread running status
    pthread_mutex_lock(&mutex);
    printf("%d,%d is working\n", first, second);
    pthread_mutex_unlock(&mutex);

    for (i = 0; i < A.cols; i++)
        res += (A.x + first * A.cols + i)->x * (B.x + i * B.cols + second)->x;
    return res;
}

void *MultWork(void *param)
{
    while (1)
    {
        int firstNum;
        int secondNum;
        int res, i, j, flag = 0, close = 0;

        pthread_mutex_lock(&mutex);
        for (i = 0; i < _C.rows; i++)
        {
            for (j = 0; j < _C.cols; j++)
            {
                if ((_C.x + i * _C.cols + j)->x == 0.0)
                {
                    firstNum = i;
                    secondNum = j;
                    (_C.x + i * _C.cols + j)->x = 1;
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
        res = CalcuOneUnit(firstNum, secondNum);
        (C.x + firstNum * C.cols + secondNum)->x = res;
    }
    pthread_exit(NULL);
}

// the main function that invokes itself at runtime, ofcourse!
void main(int argc, char *argv[])
{
    getArguments(argc, argv);

    FILE *fp1, *fp2 = NULL;
    int row, col;
    float matval = 0.0;
    int c;

    int rows_in_matrixA, cols_in_matrixA, rows_in_matrixB, cols_in_matrixB, rows_in_matrixC, cols_in_matrixC;

    char *matrixA_filename = "Mat_A.txt";
    char *matrixB_filename = "Mat_B.txt";

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

        //////////////////// TEST PURPOSES ONLY //////////////////////////
        // Expected output (from the file `Mat2.txt`) >>> Rows: 11, Columns: 8
        // printf("\nmatB.rows >>> %d", *(q + 0));
        // printf("\nmatB.columns >>> %d", *(q + 1));
        //////////////////// TEST PURPOSES ONLY //////////////////////////

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
            // float *output_matrix_ptr = (float *)malloc(matC_elements * sizeof(float));

            srand((unsigned)time(0));

            matrix target;
            target.rows = rows_in_matrixA;
            target.cols = cols_in_matrixA;
            target.x = (unit *)malloc(row * col * sizeof(unit));

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
                    (target.x + row * target.cols + col)->x = *(matrixA_ptr + counter_matA);
                    counter_matA++;
                }
            }

            printf("\nMatrix A elements >>> \n");
            A = target;
            show_matrix(A);

            // free(target.x);
            target.rows = rows_in_matrixB;
            target.cols = cols_in_matrixB;
            // target.x = (unit *)malloc(row * col * sizeof(unit));

            // A = create_matrix(rows_in_matrixA, cols_in_matrixA, fp1, &matrixA_ptr);

            // Scanning the file and storing the matrix B data in allocated memory
            int counter_matB = 0;
            for (row = 0; row < rows_in_matrixB; row++)
            {
                for (col = 0; col < cols_in_matrixB; col++)
                {
                    fscanf(fp2, "%f,", matrixB_ptr + counter_matB);
                    // printf("R: %d, C: %d, %f  ", row, col, *(matrixA_ptr + col));
                    (target.x + row * target.cols + col)->x = *(matrixB_ptr + counter_matB);
                    counter_matB++;
                }
            }

            printf("\nMatrix B elements >>> \n");
            B = target;
            show_matrix(B);

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

            // B = create_matrix(rows_in_matrixB, cols_in_matrixB, fp2, &matrixB_ptr);

            ////////////////////// MATRIX ELEMENTS OUTPUT ////////////////////////
            /*
            printing the elements present in matrix A allocated in the dynamic memory
            int count = 1, format_brackets = 1;
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
            */
            ////////////////////// MATRIX ELEMENTS OUTPUT ////////////////////////

            /* 
            // algorithm created to multiply Matrix A and Matrix B resulting in an output as Matrix C
            
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
            */

            // preparing the slicelist
            /*
            int sliceList[threadCount];
            int remainder = rows_in_matrixA % threadCount;

            // to store the sliced/divided number of matrix rows to be processed by each thread
            for (int i = 0; i < threadCount; i++)
            {
                sliceList[i] = rows_in_matrixA / threadCount;
            }

            // to update the sliced rows to be processed by each thread
            // such that none of the rows remains unprocessed/unchecked
            for (int j = 0; j < remainder; j++)
            {
                sliceList[j] = sliceList[j] + 1;
            }

            int startList[threadCount];
            int endList[threadCount];
            */

            /*
            * For threads = 3,
            * each thread will compute a new row of the matrix
            *  */
            /*
            printf("thread_count: %d\n", threadCount);
            for (int k = 0; k < threadCount; k++)
            {
                if (k == 0)
                {
                    startList[k] = 0;
                }
                else
                {
                    startList[k] = endList[k - 1] + 1;
                }

                endList[k] = startList[k] + sliceList[k] - 1;

                printf("\nstartList[%d] = %d\t\tendList[%d] = %d", k, startList[k], k, endList[k]);
            }
            */

            /*
            struct threadInfo threadDetails[threadCount];

            for (int l = 0; l < threadCount; l++)
            {
                threadDetails[l].start = startList[l];
                threadDetails[l].end = endList[l];
            }
            */

            pthread_t thread_id[threadCount];

            // struct args_struct args;

            /*
            __args = malloc(sizeof(struct args_struct) * 1);
            __args->cols_matA = cols_in_matrixA;
            __args->rows_matA = rows_in_matrixA;
            __args->cols_matB = cols_in_matrixB;
            __args->rows_matB = rows_in_matrixB;
            __args->matC_elements = matC_elements;
            __args->matrixA_ptr = &matrixA_ptr;
            __args->matrixB_ptr = &matrixB_ptr;

            for (int i = 0; i < rows_in_matrixA * cols_in_matrixA; i++)
            {
                printf("matrixA_elements: %f\t", **(__args->matrixA_ptr + i));
            }

            sem_init(&sem, 0, 1);

            printf("\n\nCreating threads and computing the matrix multiplication...\n");

            */

            pthread_mutex_init(&mutex, NULL);

            for (int m = 0; m < threadCount; m++)
            {
                /*
                // args.matC_elements = matC_elements;
                // args.cols_matA = cols_in_matrixA;
                // args.rows_matA = rows_in_matrixA;
                // args.cols_matB = cols_in_matrixB;
                // args.rows_matB = rows_in_matrixB;
                // args.threadDetails[m] = threadDetails[m];

                __args->start = threadDetails[m].start;
                __args->end = threadDetails[m].end;
                

                // pthread_create(&thread_id[m], NULL, &multiply_matrices, (void *)&args);
                pthread_create(&thread_id[m], NULL, &multiply_matrices, (void *)&__args);
                */
                pthread_create(&thread_id[m], NULL, MultWork, NULL);
            }

            for (int n = 0; n < threadCount; n++)
            {
                pthread_join(thread_id[n], NULL);
            }

            printf("\nOutput matrix C elements >>> \n");
            show_matrix(C);

            //////////////// TEST PURPOSES ONLY /////////////////////
            // for (int i = 0; i <= matC_elements; i++)
            // {
            //     printf("\noutput_matrix[%d] = %f", i, output_matrix_array[i]);
            // }
            // printf("\n\n");
            //////////////// TEST PURPOSES ONLY /////////////////////

            /*
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
            */

            // deallocating the memory
            free(matrixA_ptr);
            free(matrixB_ptr);
            // free(output_matrix_ptr);
            free(__args);
            free(target.x);
            // free(output_matrix_ptr);
            // sem_destroy(&sem);
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

// function to display a message explaining what and how arguments should be passed
void inputArguments(char *program_name)
{
    fprintf(stderr, "arguments should be in the order as specified:   %s <number of threads>\n", program_name);
    fprintf(stderr, "where number of threads should be > 0 and < 1000\n");
    exit(0);
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

// thread function to process the number of rows assigned to each thread
void *multiply_matrices(void *args)
{
    // sem_wait(&sem);

    // float output_matrix_array[matC_elements];

    // struct args_struct *arguments = (struct args_struct *)args;
    struct args_struct *arguments = args;

    int matC_elements = arguments->matC_elements;
    int cols_in_matrixA = arguments->cols_matA;
    int rows_in_matrixA = arguments->rows_matA;
    int cols_in_matrixB = arguments->cols_matB;
    int rows_in_matrixB = arguments->rows_matB;

    printf("\nmatC_elements in thread function: %d\n", matC_elements);
    printf("cols_in_matrixA in thread function: %d\n", cols_in_matrixA);
    printf("rows_in_matrixA in thread function: %d\n", rows_in_matrixA);
    printf("cols_in_matrixB in thread function: %d\n", cols_in_matrixB);
    printf("cols_in_matrixB in thread function: %d\n", cols_in_matrixB);

    // float **matrixA_ptr = (float **)malloc(rows_in_matrixA * cols_in_matrixA * sizeof(float **));
    // float **matrixA_ptr = arguments->matrixA_ptr;
    // float **matrixB_ptr = (float **)malloc(rows_in_matrixB * cols_in_matrixB * sizeof(float **));
    // float **matrixB_ptr = arguments->matrixB_ptr;
    //float **output_matrix_ptr = (float **)malloc(rows_in_matrixA * cols_in_matrixB * sizeof(float **));
    // float **output_matrix_ptr = arguments->output_matrix_ptr;

    for (int i = 0; i < 100; i++)
    {
        matrixA_array[i] = **(arguments->matrixA_ptr + i);
    }

    /////////////////////////////// For test purposes only //////////////////////////
    int count = 1, format_brackets = 1;
    printf("\nMatrix A elements (temp)>>> \n");
    printf("[  ");
    for (int i = 0; i < rows_in_matrixA * cols_in_matrixA; i++)
    {
        // printf("%f  ", **(matrixA_ptr + i));
        // printf("%f  ", **(arguments->matrixA_ptr + i));
        printf("%f  ", matrixA_array[i]);

        count++;

        if (count == cols_in_matrixA + 1)
        {
            printf("]\n[  ");
            count = 1;
        }
    }
    printf("\n\n");
    ///////////////////////////////////////////////////////////////////////////////////

    int startLimit = arguments->start;
    int endLimit = arguments->end;

    int cols_in_matrixC = cols_in_matrixB;
    int ptr_pos;
    // float *sum_ptr;

    for (int i = startLimit; i <= endLimit; i++)
    {
        for (int j = 0; j < cols_in_matrixB; j++)
        {
            float sum = 0.0;
            for (int k = 0; k < rows_in_matrixB; k++)
            {
                // sum = sum + **(matrixA_ptr + (i * cols_in_matrixA + k)) * **(matrixB_ptr + (k * cols_in_matrixB + j));
                sum = sum + **(arguments->matrixA_ptr + (i * cols_in_matrixA + k)) * **(arguments->matrixB_ptr + (k * cols_in_matrixB + j));
            }

            // sum_ptr = &sum;
            printf("SUM: %f\t", sum);
            // (output_matrix_ptr + (i * 11 + j)) = sum;
            //output_matrix_array[i * cols_in_matrixC + j] = sum; // 3 is the number of columns in matrix C
            ptr_pos = i * cols_in_matrixC + j;
            // output_matrix_ptr + (i * cols_in_matrixC + j) = &sum_ptr;
        }
        printf("\n");
    }

    // printing the output matrix C stored inside an arrayprintf("\nOutput matrix C elements >>> \n");
    // printf("[  ");
    // for (int i = 0; i < matC_elements; i++)
    // {
    //     printf("%f  ", output_matrix_array[i]);
    //     // count++;

    //     // if (count == cols_in_matrixC + 1)
    //     // {
    //     //     printf("]\n[  ");
    //     //     count = 1;
    //     // }
    // }
    // printf("  ]\n");
    // int count = 1;
    // printf("\nOutput matrix C elements >>> \n");
    // printf("[  ");
    // for (int i = 0; i < matC_elements; i++)
    // {
    //     printf("%f  ", output_matrix_array[i]);
    //     // count++;

    //     // if (count == cols_in_matrixC + 1)
    //     // {
    //     //     printf("]\n[  ");
    //     //     count = 1;
    //     // }
    // }
    // printf("  ]\n");
    // free(matrixA_ptr);
    // free(matrixB_ptr);
    // free(output_matrix_ptr);

    // sem_post(&sem);

    pthread_exit(NULL);
}
