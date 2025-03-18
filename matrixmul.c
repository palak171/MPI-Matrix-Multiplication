#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define N 70  // Matrix size

void multiply(int rank, int size) {
    double A[N][N], B[N][N], C[N][N];
    int rows_per_proc = N / size;
    
    if (rank == 0) {
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                A[i][j] = rand() % 10;
                B[i][j] = rand() % 10;
                C[i][j] = 0;
            }
    }

    MPI_Bcast(B, N*N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatter(A, rows_per_proc * N, MPI_DOUBLE, A, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < rows_per_proc; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                C[i][j] += A[i][k] * B[k][j];

    MPI_Gather(C, rows_per_proc * N, MPI_DOUBLE, C, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0) printf("Matrix multiplication completed.\n");
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();
    multiply(rank, size);
    double run_time = MPI_Wtime() - start_time;

    if (rank == 0) printf("Execution Time: %lf seconds\n", run_time);

    MPI_Finalize();
    return 0;
}
