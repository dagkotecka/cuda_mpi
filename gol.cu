#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define DEBUG 0

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

#define XY(X, Y, columnLen) (((X) * columnLen) + (Y))

#define valid(X, Y, columnLen, rowLen) (((X) < 0 || (X) >= rowLen || (Y) < 0 || (Y) >= columnLen) ? 0 : 1)

void check_error(cudaError_t error, error_case place)
{
    if (error != cudaSuccess)
    {
        switch (place)
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
calculate_new_status(const char *board, char *new_board, unsigned int columnLen, unsigned int rowLen, unsigned int threads)
{
    int aa = blockDim.x * blockIdx.x + threadIdx.x;

    while (aa < columnLen*rowLen)
    {
        int ii = aa / columnLen;
        int jj = aa % columnLen;
        int alive_neighbours = 0;

        if ((ii != 0) || (ii != (columnLen - 1)))
        {
            alive_neighbours += (valid(ii - 1, jj - 1, columnLen, rowLen) ? board[XY(ii - 1, jj - 1, columnLen)] : 0) ? 1 : 0;
            alive_neighbours += (valid(ii - 1, jj, columnLen, rowLen) ? board[XY(ii - 1, jj, columnLen)] : 0) ? 1 : 0;
            alive_neighbours += (valid(ii - 1, jj + 1, columnLen, rowLen) ? board[XY(ii - 1, jj + 1, columnLen)] : 0) ? 1 : 0;
            alive_neighbours += (valid(ii, jj - 1, columnLen, rowLen) ? board[XY(ii, jj - 1, columnLen)] : 0) ? 1 : 0;
            alive_neighbours += (valid(ii, jj + 1, columnLen, rowLen) ? board[XY(ii, jj + 1, columnLen)] : 0) ? 1 : 0;
            alive_neighbours += (valid(ii + 1, jj - 1, columnLen, rowLen) ? board[XY(ii + 1, jj - 1, columnLen)] : 0) ? 1 : 0;
            alive_neighbours += (valid(ii + 1, jj, columnLen, rowLen) ? board[XY(ii + 1, jj, columnLen)] : 0) ? 1 : 0;
            alive_neighbours += (valid(ii + 1, jj + 1, columnLen, rowLen) ? board[XY(ii + 1, jj + 1, columnLen)] : 0) ? 1 : 0;

#if DEBUG
            new_board[XY(ii, jj, columnLen)] = alive_neighbours;
#else
            char tmp_cell_val = 0;

            if (board[XY(ii, jj, columnLen)] == 0)
            {
                if (alive_neighbours == 3)
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
                if (alive_neighbours < 2)
                {
                    tmp_cell_val = 0;
                }
                else if (alive_neighbours > 3)
                {
                    tmp_cell_val = 0;
                }
                else
                {
                    tmp_cell_val = 1;
                }
            }

            new_board[XY(ii, jj, columnLen)] = tmp_cell_val;
#endif // DEBUG
        }
        aa += threads;
    }
}

int setGPUDeviceAndThreads() {
    int devicesQuantity, max_multiprocessors = 0, deviceGPU = 0;

    cudaGetDeviceCount(&devicesQuantity);

    for (int i = 0; i < devicesQuantity; i++) {
        cudaDeviceProp properties;
        cudaGetDeviceProperties(&properties, i);
        
        if (max_multiprocessors < properties.multiProcessorCount) {
            max_multiprocessors = properties.multiProcessorCount;
            deviceGPU = i;
        }
    }
    cudaSetDevice(deviceGPU);
    cudaDeviceProp properties;
    cudaGetDeviceProperties(&properties, deviceGPU);
    return properties.maxThreadsPerBlock;
}

extern "C" void cudaCalculate(char * cells, unsigned int columnLen, unsigned int rowLen)
{
    if (columnLen < 3 || rowLen < 3) exit(EXIT_FAILURE);

    int THREADS_PER_BLOCK = setGPUDeviceAndThreads();
    int BLOCKS_PER_GRID = 256;

    cudaError_t error = cudaSuccess;
    srand(time(NULL));

    char *d_board = NULL;
    char *d_new_board = NULL;

    error = cudaMalloc((void **)&d_board, columnLen*rowLen*sizeof(char));
    check_error(error, e_malloc);

    error = cudaMalloc((void **)&d_new_board, columnLen*rowLen*sizeof(char));
    check_error(error, e_malloc);

    error = cudaMemcpy(d_board, cells, columnLen*rowLen*sizeof(char), cudaMemcpyHostToDevice);
    check_error(error, e_memcpyHtD);

    calculate_new_status << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(d_board, d_new_board, columnLen, rowLen, (unsigned int) THREADS_PER_BLOCK * BLOCKS_PER_GRID);
    error = cudaGetLastError();
    check_error(error, e_kernel);
    cudaDeviceSynchronize();

    error = cudaMemcpy(cells, d_new_board, columnLen*rowLen*sizeof(char), cudaMemcpyDeviceToHost);
    check_error(error, e_memcpyDtH);

    error = cudaFree(d_board);
    check_error(error, e_dealloc);

    error = cudaFree(d_new_board);
    check_error(error, e_dealloc);

    error = cudaDeviceReset();
    check_error(error, e_reset);

    return;
}