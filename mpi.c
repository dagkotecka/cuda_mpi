#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define DATA 0

unsigned int* cudaCalculate(unsigned int * cells, unsigned int columnLen, unsigned int rowLen);

void print_board(unsigned int * board, unsigned int columnLen, unsigned int rows, unsigned int printing)
{
    unsigned int ii = 0;
    unsigned int jj = 0;

    if(! printing)
        return;

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
    unsigned int intervalSize, intervalUints, fullIntervalUints;
    unsigned int *board = NULL; 
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
    intervalUints = intervalSize * columnLen;
    fullIntervalUints = (intervalSize + 2) * columnLen;

    rows += 2; // due to first and last zeros row
    //printf("rows %u, columnLen %u, cycles %u\n", rows, columnLen, cycles);

    //printf("Starting nodes...\n");
    if(myId == 0) // master
    {
        unsigned int ii = 0;
        unsigned int jj = 0;
        unsigned int *new_ptr = NULL;

        board = (unsigned int *) calloc (rows * columnLen, sizeof(unsigned int));

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

        print_board(board, columnLen, rows, 1);

        while(cycles--)
        {
            for(ii = 0; ii < (unsigned)slaves; ii++)
            {
                new_ptr = &board [ii * (intervalUints)];
                MPI_Send(new_ptr, fullIntervalUints, MPI_UNSIGNED, ii + 1, DATA, MPI_COMM_WORLD);
            }

            for(ii = 0; ii < (unsigned)slaves; ii++)
            {
                new_ptr = &board[ii * intervalUints + columnLen];
                MPI_Recv(new_ptr, intervalUints, MPI_UNSIGNED, ii + 1, DATA, MPI_COMM_WORLD, &status);
            }

            for(jj = 0; jj < columnLen; ++jj)
                board[jj] = 0;

            for(ii = columnLen + 1; ii < rows; ++ii)
                for(jj = 0; jj < columnLen; ++jj)
                    board[ii*columnLen + jj] = 0;

            print_board(board, columnLen, rows, printing);
        }

        printf("Processing node %d done!\n", myId);
    }
    else // slaves
    {
        unsigned int *new_ptr = NULL;
        unsigned int *slaveBoard = (unsigned int *) malloc (fullIntervalUints * sizeof(unsigned int));
        //printf("Processing node %d...\n", myId);

        if(slaveBoard == NULL)
        {
            printf("Malloc failed!\n");
            return 0;
        }
        while(cycles--)
        {
            MPI_Recv(slaveBoard, fullIntervalUints, MPI_UNSIGNED, 0, DATA, MPI_COMM_WORLD, &status);

            //printf("Processing node %d...2\n", myId);
            cudaCalculate(slaveBoard, columnLen, fullIntervalUints / columnLen);
            //printf("Processing node %d...3\n", myId);
            new_ptr = &slaveBoard[columnLen];
            MPI_Send(new_ptr, intervalUints, MPI_UNSIGNED, 0, DATA, MPI_COMM_WORLD);
        }
        free(slaveBoard);
        slaveBoard = NULL;
        printf("Processing node %d done!\n", myId);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if(myId == 0)
    {
        print_board(board, columnLen, rows, 1);
        free(board);
        board = NULL;
    }

    MPI_Finalize();
    //printf("Everything's fine. Closing now.\n");
    return 0;
}
