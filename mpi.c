#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define DATA 0

void cudaCalculate(char * cells, unsigned int columnLen, unsigned int rowLen);

void print_board(char * board, unsigned int columnLen, unsigned int rows)
{
    unsigned int ii = 0;
    unsigned int jj = 0;

    ii = system("clear"); // due to warning of unused return

    for(ii = 0; ii < rows; ii++)
    {
        for(jj = 0; jj < columnLen; jj++)
        {
            if(board[columnLen * ii + jj] == 0)
                printf(" ");
            else
                printf("#");
        }
        printf("\n");
    }
    printf("\n");
}

int main(int argc, char **argv) 
{
    int myId, procs, slaves;
    unsigned int columnLen, cycles, rows, printing;
    unsigned int intervalSize, intervalChars, fullIntervalChars;
    char *board = NULL; 
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Comm_size(MPI_COMM_WORLD, &procs); 

    //procs = 4;
    //myId = 0;

    if(procs == 1)
    {
        printf("Needs at least 1 slave.\n");
        return 0;
    }

    if(myId == 0)
        printf("Starting MPI...\n");

    srand ((unsigned int) time(NULL));
    slaves = procs - 1;
    //printf("Slaves: %d\n", slaves);

    if(argc >= 4)
    {
        rows = columnLen = (unsigned int) atoi(argv[1]);
        cycles = (unsigned int) atoi(argv[2]);
        printing = (unsigned int) atoi(argv[3]);
    }
    else
    {
        rows = columnLen = 16;
        cycles = 10;
        printing = 1;
    }

    if(columnLen % slaves != 0)
    {
        rows += slaves - (columnLen % slaves); // addition to be slave multiple
    }

    intervalSize = rows / slaves;
    intervalChars = intervalSize * columnLen;
    fullIntervalChars = (intervalSize + 2) * columnLen;

    rows += 2; // due to first and last zeros row
    //printf("rows %u, columnLen %u, cycles %u\n", rows, columnLen, cycles);

    //printf("Starting nodes...\n");
    if(myId == 0) // master
    {
        unsigned int ii = 0;
        unsigned int jj = 0;
        char *new_ptr = NULL;

        board = (char *) calloc (rows * columnLen, sizeof(char));

        if(board == NULL)
        {
            printf("Calloc failed!\n");
            return 0;
        }

        for(ii = 1; ii <= columnLen; ++ii)
        {
            for(jj = 0; jj < columnLen; ++jj)
            {
                board[(ii*columnLen) + jj] = rand() % 2;// (ii) * 100 + jj + 1;
            }
        }

        if(printing >= 1)
            print_board(board, columnLen, rows);

        while(cycles--)
        {
            for(ii = 0; ii < (unsigned)slaves; ii++)
            {
                new_ptr = &board [ii * (intervalChars)];
                MPI_Send(new_ptr, fullIntervalChars, MPI_CHAR, ii + 1, DATA, MPI_COMM_WORLD);
            }

            for(ii = 0; ii < (unsigned)slaves; ii++)
            {
                new_ptr = &board[ii * intervalChars + columnLen];
                MPI_Recv(new_ptr, intervalChars, MPI_CHAR, ii + 1, DATA, MPI_COMM_WORLD, &status);
            }

            for(jj = 0; jj < columnLen; ++jj)
                board[jj] = 0;

            for(ii = columnLen + 1; ii < rows; ++ii)
                for(jj = 0; jj < columnLen; ++jj)
                    board[ii*columnLen + jj] = 0;

            if(printing > 1)
                print_board(board, columnLen, rows);
        }

        printf("Processing node %d done!\n", myId);
    }
    else // slaves
    {
        char *new_ptr = NULL;
        char *slaveBoard = (char *) malloc (fullIntervalChars * sizeof(char));
        //printf("Processing node %d...\n", myId);

        if(slaveBoard == NULL)
        {
            printf("Malloc failed!\n");
            return 0;
        }
        while(cycles--)
        {
            MPI_Recv(slaveBoard, fullIntervalChars, MPI_CHAR, 0, DATA, MPI_COMM_WORLD, &status);

            //printf("Processing node %d...2\n", myId);
            cudaCalculate(slaveBoard, columnLen, fullIntervalChars / columnLen);
            //printf("Processing node %d...3\n", myId);
            new_ptr = &slaveBoard[columnLen];
            MPI_Send(new_ptr, intervalChars, MPI_CHAR, 0, DATA, MPI_COMM_WORLD);
        }
        free(slaveBoard);
        slaveBoard = NULL;
        printf("Processing node %d done!\n", myId);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(myId == 0)
    {
        if(printing >= 1)
            print_board(board, columnLen, rows);
        free(board);
        board = NULL;
    }

    MPI_Finalize();
    //printf("Everything's fine. Closing now.\n");
    return 0;
}
