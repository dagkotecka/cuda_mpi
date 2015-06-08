#include <stdio.h>
#include <cuda_runtime.h>
#include <cstdlib>

#define DEBUG 0

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

#define XY(X, Y, columnLen) (((X) * columnLen) + (Y))

#define valid(X, Y, columnLen, rowLen) (((X) < 0 || (X) >= columnLen || (Y) < 0 || (Y) >= rowLen) ? 0 : 1)

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
calculate_new_status(const unsigned int *board, unsigned int *new_board, unsigned int columnLen, unsigned int rowLen)
{
	int aa = blockDim.x * blockIdx.x + threadIdx.x;

	if (aa < columnLen*rowLen)
	{
		int ii = aa / columnLen;
		int jj = aa % rowLen;
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
			unsigned int tmp_cell_val = 0;

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
	}
}

extern "C" unsigned int* cudaCalculate(unsigned int * cells, unsigned int columnLen, unsigned int rowLen)
{
	if (columnLen < 3 || rowLen < 3) exit(EXIT_FAILURE);

	int THREADS_PER_BLOCK = 16;
	int BLOCKS_PER_GRID = (columnLen*rowLen + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	cudaError_t error = cudaSuccess;
	srand(time(NULL));

	unsigned int *d_board = NULL;
	unsigned int *d_new_board = NULL;

	error = cudaMalloc((void **)&d_board, columnLen*rowLen*sizeof(unsigned int));
	check_error(error, e_malloc);

	error = cudaMalloc((void **)&d_new_board, columnLen*rowLen*sizeof(unsigned int));
	check_error(error, e_malloc);

	error = cudaMemcpy(d_board, cells, columnLen*rowLen*sizeof(unsigned int), cudaMemcpyHostToDevice);
	check_error(error, e_memcpyHtD);

	cudaDeviceSynchronize();
	calculate_new_status << <BLOCKS_PER_GRID, THREADS_PER_BLOCK >> >(d_board, d_new_board, columnLen, rowLen);
	cudaDeviceSynchronize();
	error = cudaGetLastError();
	check_error(error, e_kernel);

	error = cudaMemcpy(cells, d_new_board, columnLen*rowLen*sizeof(unsigned int), cudaMemcpyDeviceToHost);
	check_error(error, e_memcpyDtH);

	error = cudaFree(d_board);
	check_error(error, e_dealloc);

	error = cudaFree(d_new_board);
	check_error(error, e_dealloc);

	error = cudaDeviceReset();
	check_error(error, e_reset);

	return cells;
}