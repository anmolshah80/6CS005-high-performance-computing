#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <pthread.h>

typedef struct
{
    int x;
} unit;

typedef struct
{
    int row;
    int col;
    unit *x;
} matrix;

matrix A, B, C, _C;
pthread_mutex_t mutex;

matrix target;

// function to find the number of rows and columns of each matrix
int *find_number_of_rows_and_columns(const char *file_name, FILE *fp)
{
    FILE *fp1, *fp2 = NULL;
    fp = fopen(file_name, "r");
    int newRows = 1;
    int newCols = 1;
    char ch;

    // static int rows_cols[10];

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

    if (fp == fp1)
    {
        A.row = newRows;
        A.col = newCols;
    }
    else if (fp == fp2)
    {
        B.row = newRows;
        B.col = newCols;
    }

    // rows_cols[0] = newRows;
    // rows_cols[1] = newCols;

    // printf("\nRows: %d, Cols: %d\n", rows_cols[0], rows_cols[1]);

    // return rows_cols;
}

matrix CreatMatrix(int row, int col, const char *filename) //Row is the number of rows, col is the number of columns, that is, the matrix is row*col
{
    // matrix target;
    int i, j;
    float data = 0.0;
    // int matrix_elements = row * col;
    // float *matrix_ptr = (float *)malloc(matrix_elements * sizeof(float));

    //target = (matrix *)malloc(sizeof(matrix));

    target.row = row;
    target.col = col;
    target.x = (unit *)malloc(row * col * sizeof(unit));

    if (filename != NULL)
    {
        FILE *fp = fopen(filename, "r");
        for (i = 0; i < row; i++)
            for (j = 0; j < col; j++)
            {
                // data = rand() % 100;
                fscanf(fp, "%f,", &data);

                (target.x + i * target.col + j)->x = data;
            }
    }
    // else
    // {
    //     for (i = 0; i < row; i++)
    //         for (j = 0; j < col; j++)
    //         {
    //             // data = rand() % 100;
    //             fscanf(fp, "%f,", &data);

    //             (target.x + i * target.col + j)->x = data;
    //         }
    // }

    return target;
}

int CalcuOneUnit(int first, int second)
{
    int i, res = 0;

    pthread_mutex_lock(&mutex);
    printf("%d,%d is working\n", first, second);
    pthread_mutex_unlock(&mutex); // Here for testing thread running status

    for (i = 0; i < A.col; i++)
        res += (A.x + first * A.col + i)->x * (B.x + i * B.col + second)->x;
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
        for (i = 0; i < _C.row; i++)
        {
            for (j = 0; j < _C.col; j++)
            {
                if ((_C.x + i * _C.col + j)->x == NULL)
                {
                    firstNum = i;
                    secondNum = j;
                    (_C.x + i * _C.col + j)->x = 1;
                    close = 1;
                    break;
                }
            }
            if (close == 1)
                break;
            else if (i == _C.row - 1)
                flag = 1;
        }
        pthread_mutex_unlock(&mutex);

        if (flag == 1)
            pthread_exit(NULL);
        res = CalcuOneUnit(firstNum, secondNum);
        (C.x + firstNum * C.col + secondNum)->x = res;
    }
    pthread_exit(NULL);
}

void ShowMatrix(matrix shows)
{
    int row = shows.row;
    int col = shows.col;
    int i, j;
    for (i = 0; i < row; i++)
    {
        for (j = 0; j < col; j++)
            printf("%d\t", (shows.x + i * col + j)->x);
        printf("\n");
    }
}

int main()
{
    int row, col;
    int i;
    pthread_t thread1, thread2, thread3, thread4;
    printf("Simple example \n");

    srand((unsigned)time(NULL));
    // printf("input row and col of Matrix A :\n");
    // scanf("%d %d", &row, &col);

    char *matrixA_filename = "MatA.txt";
    char *matrixB_filename = "MatB.txt";

    FILE *fp, *fp1, *fp2 = NULL;

    fp1 = fopen(matrixA_filename, "r");
    fp2 = fopen(matrixB_filename, "r");

    if (fp1 != NULL && fp2 != NULL)
    {
        printf("Matrix A is :\n");
        find_number_of_rows_and_columns(matrixA_filename, fp1);
        A = CreatMatrix(A.row, A.col, matrixA_filename);
        ShowMatrix(A);

        printf("Matrix B is :\n");
        find_number_of_rows_and_columns(matrixB_filename, fp2);
        B = CreatMatrix(B.row, B.col, matrixB_filename);
        ShowMatrix(B);
    }

    // printf("input row and col of Matrix B :\n");
    // scanf("%d %d", &row, &col);

    if (A.col != B.row)
    {
        printf("error input");
        return 1;
    }

    C = CreatMatrix(A.row, B.col, matrixA_filename);
    for (i = 0; i < C.col * C.row; i++)
        (C.x + i)->x = NULL;

    _C = CreatMatrix(A.row, B.col, matrixB_filename);
    for (i = 0; i < _C.col * _C.row; i++)
        (_C.x + i)->x = NULL;

    pthread_mutex_init(&mutex, NULL);
    pthread_create(&thread1, NULL, MultWork, NULL);
    pthread_create(&thread2, NULL, MultWork, NULL);
    pthread_create(&thread3, NULL, MultWork, NULL);
    pthread_create(&thread4, NULL, MultWork, NULL);
    pthread_join(thread1, NULL);
    pthread_join(thread2, NULL);
    pthread_join(thread3, NULL);
    pthread_join(thread4, NULL);

    printf("Matrix C is :\n");
    ShowMatrix(C);

    free(target.x);
    return 0;
}