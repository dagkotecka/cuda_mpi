echo "building...";

echo "cuda gol"
nvcc -arch=compute_20 -c gol.cu -o gol.o
echo "mpi"
/opt/mpich/ch-p4/bin/mpicc -c mpi.c -o mpi.o -lm
echo "mpi cuda"
/opt/mpich/ch-p4/bin/mpicc mpi.o gol.o -o mpi_gol -L/usr/local/cuda/targets/x86_64-linux/lib -L/usr/lib64/ -lcudart -lstdc++
rm gol.o mpi.o