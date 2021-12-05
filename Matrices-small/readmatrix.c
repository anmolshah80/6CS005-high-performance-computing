#include <stdio.h>

int rows, columns;

void find_number_of_rows_and_columns(const char *file_name);

void main()
{
    FILE *fp = NULL;
    int row, col;
    float matval = 0.0;
    int c;

    fp = fopen("Mat2.txt", "r");

    if (fp != NULL)
    {
        find_number_of_rows_and_columns("Mat2.txt");

        // Expected output (from the file `Mat1.txt`) >>> Rows: 8, Columns: 11
        printf("Rows: %d, Columns: %d\n", rows, columns);

        for (row = 0; row < rows; row++)
        {
            printf("[ ");
            for (col = 0; col < columns - 1; col++)
            {
                fscanf(fp, "%f,", &matval);
                printf("%f\t", matval);
            }
            fscanf(fp, "%f", &matval);
            printf("%f ]\n", matval);
        }
        fclose(fp);
    }
    else
    {
        printf("\nNo such file found!\n");
    }
}

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