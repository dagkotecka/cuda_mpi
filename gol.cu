#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>

int SIZE = 16;
int LIFE_CYCLES = 10;

#define DEBUG 0

typedef int cell_val; 
typedef unsigned int uint;

enum error_case
{
    e_malloc,
    e_memcpyHtD,
    e_memcpyDtH,
    e_kernel,
    e_dealloc,
    e_reset,
    e_default = 99
};

#define XY(X, Y) (((X) * SIZE) + (Y))

#define valid(X, Y) (((X) < 0 || (X) >= SIZE || (Y) < 0 || (Y) >= SIZE) ? 0 : 1)

void print_board(cell_val * board)
{
#if ! DEBUG
    std::system("clear");
#endif
    for(uint ii = 0; ii < SIZE; ii++)
    {
        for(uint jj = 0; jj < SIZE; jj++)
        {
#if DEBUG
            printf("%d", board[XY(ii, jj)]);
#else
            if(board[XY(ii, jj)] == 0)
                printf(" ");
            else
                printf("#");
#endif // DEBUG
        }
        printf("\n");
    }
    printf("\n");
}

void check_error(cudaError_t error, error_case place)
{
    if (error != cudaSuccess)
    {
        switch(place)
        {
        case e_malloc:
            fprintf(stderr, "ERROR! CUDA malloc failed (error code %s)\n", cudaGetErrorString(error));
            break;
        case e_memcpyHtD:
            fprintf(stderr, "ERROR! CUDA memcpy to device failed (error code %s)\n", cudaGetErrorString(error));
            break;
        case e_memcpyDtH:
            fprintf(stderr, "ERROR! CUDA memcpy to host failed (error code %s)\n", cudaGetErrorString(error));
            break;
        case e_kernel:
            fprintf(stderr, "ERROR! CUDA kernel function failed (error code %s)\n", cudaGetErrorString(error));
            break;
        case e_dealloc:
            fprintf(stderr, "ERROR! CUDA deallocation failed (error code %s)\n", cudaGetErrorString(error));
            break;
        case e_reset:
            fprintf(stderr, "ERROR! CUDA failed when tried to reset (error code %s)\n", cudaGetErrorString(error));
            break;
        default:
            fprintf(stderr, "ERROR! CUDA unrecognized error (error code %s)\n", cudaGetErrorString(error));
            break;
        }
        exit(EXIT_FAILURE);
    }
}

__global__ void
calculate_new_status(const cell_val *board, cell_val *new_board, const int SIZE)
{
    int aa = blockDim.x * blockIdx.x + threadIdx.x;

    if(aa < SIZE*SIZE)
    {
        int ii = aa / SIZE;
        int jj = aa % SIZE;
        int alive_neighbours = 0;

        alive_neighbours += (valid(ii-1, jj-1) ? board[XY(ii-1, jj-1)] : 0)?1:0;
        alive_neighbours += (valid(ii-1, jj  ) ? board[XY(ii-1, jj  )] : 0)?1:0;
        alive_neighbours += (valid(ii-1, jj+1) ? board[XY(ii-1, jj+1)] : 0)?1:0;
        alive_neighbours += (valid(ii,   jj-1) ? board[XY(ii,   jj-1)] : 0)?1:0;
        alive_neighbours += (valid(ii,   jj+1) ? board[XY(ii,   jj+1)] : 0)?1:0;
        alive_neighbours += (valid(ii+1, jj-1) ? board[XY(ii+1, jj-1)] : 0)?1:0;
        alive_neighbours += (valid(ii+1, jj  ) ? board[XY(ii+1, jj  )] : 0)?1:0;
        alive_neighbours += (valid(ii+1, jj+1) ? board[XY(ii+1, jj+1)] : 0)?1:0;

#if DEBUG
        new_board[XY(ii, jj)] = alive_neighbours;
#else
        cell_val tmp_cell_val = 0;
    
        if(board[XY(ii,jj)] == 0)
        {
            if(alive_neighbours == 3)
            {
                tmp_cell_val = 1;
            }
            else
            {
                tmp_cell_val = 0;
            }
        }
        else
        {
            if(alive_neighbours < 2)
            {
                tmp_cell_val = 0;
            }
            else if(alive_neighbours > 3)
            {
                tmp_cell_val = 0;
            }
            else
            {
                tmp_cell_val = 1;
            }
        }

        new_board[XY(ii,jj)] = tmp_cell_val;
#endif // DEBUG
    }
}

int main(int argc, char *argv[])
{
    if(argc < 3)
    {
        printf("Board size: %dx%d. Cycles: %d\n", SIZE, SIZE, LIFE_CYCLES);
    }
    else
    {
        SIZE = atoi(argv[1]);
        LIFE_CYCLES = atoi(argv[2]);
        printf("Board size: %dx%d. Cycles: %d\n", SIZE, SIZE, LIFE_CYCLES);
    }
    
    int THREADS_PER_BLOCK = 16;
    int BLOCKS_PER_GRID =  (SIZE*SIZE + THREADS_PER_BLOCK -1) / THREADS_PER_BLOCK;
    
    cudaError_t error = cudaSuccess;
    srand (time(NULL));

    cell_val *board = (cell_val*) malloc (SIZE * SIZE * sizeof(cell_val*));

    if(board == NULL)
    {
        fprintf(stderr, "Malloc failed!\n");
         exit(EXIT_FAILURE);
    }

    for(uint ii = 0; ii < SIZE; ii++)
    {
        for(uint jj = 0; jj < SIZE; jj++)
        {
            board[XY(ii, jj)] = (cell_val) rand() % 2;
        }
    }

    printf("Initial board\n");
    print_board(board);

    cell_val *d_board = NULL;
    cell_val *d_new_board = NULL;

    error = cudaMalloc((void **)&d_board, SIZE*SIZE*sizeof(cell_val));
    check_error(error, e_malloc);

    error = cudaMalloc((void **)&d_new_board, SIZE*SIZE*sizeof(cell_val));
    check_error(error, e_malloc);

    for(uint cc = 1; cc <= LIFE_CYCLES; cc++)
    {
        error = cudaMemcpy(d_board, board, SIZE*SIZE*sizeof(cell_val), cudaMemcpyHostToDevice);
        check_error(error, e_memcpyHtD);

        calculate_new_status<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_board, d_new_board, SIZE);
        error = cudaGetLastError();
        check_error(error, e_kernel);

        error = cudaMemcpy(board, d_new_board, SIZE*SIZE*sizeof(cell_val), cudaMemcpyDeviceToHost);
        check_error(error, e_memcpyDtH);

        printf("Board after cycle %d\n", cc);
        print_board(board);

    }
    printf("All cycles done!\n");

    error = cudaFree(d_board);
    check_error(error, e_dealloc);

    error = cudaFree(d_new_board);
    check_error(error, e_dealloc);

    error = cudaDeviceReset();
    check_error(error, e_reset);

    printf("Everything cleaned up! CUDA'ing successful!\n");
    return 0;
}