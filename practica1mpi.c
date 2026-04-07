#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

// Función para reservar memoria para una matriz contigua (más eficiente para MPI)
double* allocate_matrix_flat(int n) {
    return (double*)malloc(n * n * sizeof(double));
}

// Inicializar matriz con valores aleatorios
void init_matrix(double* m, int n) {
    for (int i = 0; i < n * n; i++) {
        m[i] = rand() % 10;
    }
}

int main(int argc, char* argv[]) {
    int rank, size;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 2) {
        if (rank == 0) printf("Uso: mpirun -np <procs> %s <tamano_matriz>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    int n = atoi(argv[1]);
    
    // El número de filas debe ser divisible por el número de procesos para simplificar
    if (n % size != 0) {
        if (rank == 0) printf("Error: El tamaño %d debe ser divisible por el número de procesos %d\n", n, size);
        MPI_Finalize();
        return 1;
    }

    int rows_per_proc = n / size;
    double *A = NULL, *B = NULL, *C = NULL;
    double *local_A = (double*)malloc(rows_per_proc * n * sizeof(double));
    double *local_C = (double*)malloc(rows_per_proc * n * sizeof(double));
    B = (double*)malloc(n * n * sizeof(double));

    if (rank == 0) {
        A = allocate_matrix_flat(n);
        C = allocate_matrix_flat(n);
        srand(time(NULL));
        init_matrix(A, n);
        init_matrix(B, n);
    }

    double start = MPI_Wtime();

    // 1. Repartir las filas de A entre todos los procesos
    MPI_Scatter(A, rows_per_proc * n, MPI_DOUBLE, local_A, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 2. Enviar la matriz B completa a todos (Broadcast)
    MPI_Bcast(B, n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 3. Multiplicación local
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < n; j++) {
            local_C[i * n + j] = 0.0;
            for (int k = 0; k < n; k++) {
                local_C[i * n + j] += local_A[i * n + k] * B[k * n + j];
            }
        }
    }

    // 4. Reunir los resultados en el proceso 0
    MPI_Gather(local_C, rows_per_proc * n, MPI_DOUBLE, C, rows_per_proc * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double end = MPI_Wtime();

    if (rank == 0) {
        printf("Tamaño: %dx%d\n", n, n);
        printf("Procesos MPI: %d\n", size);
        printf("Tiempo de ejecucion: %.4f segundos\n", end - start);

        // Checksum para validar
        double checksum = 0.0;
        for (int i = 0; i < n * n; i++) checksum += C[i];
        printf("Checksum: %.2f\n", checksum);

        free(A);
        free(C);
    }

    free(B);
    free(local_A);
    free(local_C);

    MPI_Finalize();
    return 0;
}