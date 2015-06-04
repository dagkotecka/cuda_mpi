#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#define DATA 0


void cudaCalculate(unsigned int * board, unsigned int columnLen, unsigned int rowLen)
{

}

void print_board(unsigned int * board, unsigned int columnLen, unsigned int rows)
{
	unsigned int ii = 0;
	unsigned int jj = 0;

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
    unsigned int columnLen, cycles, rows;
	unsigned int *board = NULL; 
    MPI_Status status;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &myId);
    MPI_Comm_size(MPI_COMM_WORLD, &procs); 

	if(procs == 1)
	{
		printf("Needs at least 1 slave.\n");
		return 0;
	}
    
	printf("Starting MPI...\n");

	srand (time(NULL));
	slaves = procs - 1;
	printf("Slaves: %d\n", slaves);

    if(argc >= 3)
    {
        columnLen = atoi(argv[1]);
        cycles = atoi(argv[2]);
    }
	else
	{
		rows = columnLen = 16;
        cycles = 10;
	}
    printf("rows %d, cycles %d\n", rows, cycles);

	if(columnLen % slaves != 0) 
		rows += slaves - (columnLen % slaves); // addition to be slave multiple
	rows += 2; // due to first and last zeros row

	unsigned int intervalSize = rows / slaves;
	unsigned int intervalUints = intervalSize * columnLen;
	unsigned int fullIntervalUints = (intervalSize + 2) * columnLen;

	printf("Starting nodes...\n");
	if(myId == 0) // master
	{
		int ii = 0;
		int jj = 0;

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
				board[(ii*columnLen) + jj] = rand() % 2;
			}
		}

		print_board(board, columnLen, rows);

		for(ii = 0; ii < slaves; ii++)
		{
			unsigned int *new_ptr = &board [ii * (intervalUints)];
			MPI_Send(new_ptr, fullIntervalUints, MPI_UNSIGNED, ii + 1, DATA, MPI_COMM_WORLD);
		}

		for(ii = 0; ii < slaves; ii++)
		{
			unsigned int *new_ptr = &board[ii * intervalUints + columnLen];
			MPI_Recv(new_ptr, intervalUints, MPI_UNSIGNED, ii + 1, DATA, MPI_COMM_WORLD, &status);
		}

		print_board(board, columnLen, rows);

		printf("Processing node %d done!\n", myId);
	}
	else // slaves
	{
		printf("Processing node %d...\n", myId);
		unsigned int *slaveBoard = (unsigned int *) malloc (fullIntervalUints * sizeof(unsigned int));

		if(slaveBoard == NULL)
		{
			printf("Malloc failed!\n");
			return 0;
		}

		MPI_Recv(slaveBoard, fullIntervalUints, MPI_UNSIGNED, 0, DATA, MPI_COMM_WORLD, &status);

		printf("Processing node %d...2\n", myId);
		cudaCalculate(slaveBoard, columnLen, fullIntervalUints / columnLen);
		printf("Processing node %d...3\n", myId);
		unsigned int *new_ptr = &slaveBoard[columnLen];
		MPI_Send(new_ptr, intervalUints, MPI_UNSIGNED, 0, DATA, MPI_COMM_WORLD);
		//free(slaveBoard);
		printf("Processing node %d done!\n", myId);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	//if(board != NULL)
	//	free(board);

	MPI_Finalize();
	printf("Everything's fine. Closing now.\n");
	return 0;
}
