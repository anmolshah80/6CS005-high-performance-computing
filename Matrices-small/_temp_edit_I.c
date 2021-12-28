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

int rows, columns;

matrix CreatMatrix(int row, int col) //Row is the number of rows, col is the number of columns, that is, the matrix is row*col
{
    matrix target;
    int i, j;
    int data;
    //target = (matrix *)malloc(sizeof(matrix));

    target.row = row;
    target.col = col;
    target.x = (unit *)malloc(row * col * sizeof(unit));
    for (i = 0; i < row; i++)
        for (j = 0; j < col; j++)
        {
            data = rand() % 100;
            (target.x + i * target.col + j)->x = data;
        }
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

// function to read matrix files, its rows and columns, and
// the elements present inside of it
void find_number_of_rows_and_columns(const char *file_name)
{
    FILE *fp = fopen(file_name, "r");
    int newRows = 1;
    int newCols = 1;
    char ch;

    if (fp != NULL)
    {
        while (!feof(fp))
        {
            ch = fgetc(fp);
            // printf("\nCh: %c", ch);
            if (ch == '\n')
            {
                newRows++;
                columns = newCols;
                // printf("\nRows: %d, Column: %d", rows, columns);
                newCols = 1;
            }
            else if (ch == ',')
            {
                newCols++;
            }
        }
        rows = newRows;
        // columns = newCols;
    }
    else
    {
        printf("\nNo such file found!\n");
    }
}

int main()
{
    int row, col;
    int i;
    pthread_t thread1, thread2, thread3, thread4;
    printf("Simple example \n");

    srand((unsigned)time(NULL));
    printf("input row and col of Matrix A :\n");
    scanf("%d %d", &row, &col);
    printf("Matrix A is :\n");
    A = CreatMatrix(row, col);
    ShowMatrix(A);
    printf("input row and col of Matrix B :\n");
    scanf("%d %d", &row, &col);
    printf("Matrix B is :\n");
    B = CreatMatrix(row, col);
    ShowMatrix(B);
    if (A.col != B.row)
    {
        printf("error input");
        return 1;
    }

    C = CreatMatrix(A.row, B.col);
    for (i = 0; i < C.col * C.row; i++)
        (C.x + i)->x = NULL;
    _C = CreatMatrix(A.row, B.col);
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
    return 0;
}