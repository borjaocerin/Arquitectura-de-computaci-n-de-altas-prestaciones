#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Reservar memoria para matrices rectangulares
double** allocate_matrix(int rows, int cols) {
    double** m = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        m[i] = (double*)malloc(cols * sizeof(double));
    }
    return m;
}

void free_matrix(double** m, int rows) {
    for (int i = 0; i < rows; i++) {
        free(m[i]);
    }
    free(m);
}

void init_matrix(double** m, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            m[i][j] = rand() % 10;
        }
    }
}

// Multiplicación PARALELA adaptada a M, K, N
void multiply_omp(double** A, double** B, double** C, int M, int K, int N) {
    // collapse(2) reparte las iteraciones de los dos primeros bucles entre los hilos
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < K; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {
    // Pedimos los 3 parámetros: Filas A, Intermedio, Columnas B
    if (argc != 4) {
        printf("Uso: %s <filas_A> <cols_A_filas_B> <cols_B>\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    if (M <= 0 || K <= 0 || N <= 0) {
        printf("Las dimensiones deben ser mayores que 0\n");
        return 1;
    }

    srand(time(NULL));

    double** A = allocate_matrix(M, K);
    double** B = allocate_matrix(K, N);
    double** C = allocate_matrix(M, N);

    init_matrix(A, M, K);
    init_matrix(B, K, N);

    // Medición de tiempo específica de OpenMP (más precisa para hilos)
    double start = omp_get_wtime();

    multiply_omp(A, B, C, M, K, N);

    double end = omp_get_wtime();

    printf("--- Resultado Paralelo ---\n");
    printf("Dimensiones: A(%dx%d) * B(%dx%d) -> C(%dx%d)\n", M, K, K, N, M, N);
    printf("Hilos usados: %d\n", omp_get_max_threads());
    printf("Tiempo de ejecucion: %.4f segundos\n", end - start);

    // Liberar memoria
    free_matrix(A, M);
    free_matrix(B, K);
    free_matrix(C, M);

    return 0;
}