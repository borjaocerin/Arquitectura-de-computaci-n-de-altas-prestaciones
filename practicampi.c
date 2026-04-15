#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// Función para reservar memoria para una matriz contigua (necesario para MPI_Scatter/Gather)
double* allocate_matrix_flat(int rows, int cols) {
    return (double*)malloc(rows * cols * sizeof(double));
}

void init_matrix(double* m, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        m[i] = rand() % 10;
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 4) {
        if (rank == 0) printf("Uso: mpirun -np <procs> %s <filas_A> <cols_A_filas_B> <cols_B>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int M = atoi(argv[1]); // Filas A
    int K = atoi(argv[2]); // Cols A / Filas B
    int N = atoi(argv[3]); // Cols B
    
    // El número de filas de A (M) debe ser divisible por el número de procesos
    if (M % size != 0) {
        if (rank == 0) printf("Error: El número de filas de A (%d) debe ser divisible por el número de procesos (%d)\n", M, size);
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = M / size;
    
    double *A = NULL, *B = NULL, *C = NULL;
    
    // Matrices locales para cada proceso
    double *local_A = (double*)malloc(rows_per_proc * K * sizeof(double));
    double *local_C = (double*)malloc(rows_per_proc * N * sizeof(double));
    
    // La matriz B la necesitan todos completa
    B = (double*)malloc(K * N * sizeof(double));

    if (rank == 0) {
        A = allocate_matrix_flat(M, K);
        C = allocate_matrix_flat(M, N);
        srand(time(NULL));
        init_matrix(A, M, K);
        init_matrix(B, K, N);
    }

    // Sincronizar para empezar a medir tiempo
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    // 1. Repartir las filas de A: cada proceso recibe (rows_per_proc * K) elementos
    MPI_Scatter(A, rows_per_proc * K, MPI_DOUBLE, local_A, rows_per_proc * K, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 2. Enviar la matriz B completa a todos (K * N elementos)
    MPI_Bcast(B, K * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 3. Multiplicación local: (rows_per_proc x K) * (K x N) = (rows_per_proc x N)
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++) {
            local_C[i * N + j] = 0.0;
            for (int k = 0; k < K; k++) {
                local_C[i * N + j] += local_A[i * K + k] * B[k * N + j];
            }
        }
    }

    // 4. Reunir los resultados en la matriz C del proceso 0
    MPI_Gather(local_C, rows_per_proc * N, MPI_DOUBLE, C, rows_per_proc * N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("--- Resultado MPI ---\n");
        printf("Dimensiones: A(%dx%d) * B(%dx%d) -> C(%dx%d)\n", M, K, K, N, M, N);
        printf("Procesos MPI: %d\n", size);
        printf("Tiempo de ejecucion: %.4f segundos\n", end - start);

        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}