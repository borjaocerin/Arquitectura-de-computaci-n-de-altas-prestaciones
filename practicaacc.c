#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <openacc.h>

/**
 * Reserva de memoria para una matriz plana (1D)
 */
double* allocate_matrix_flat(int rows, int cols) {
    return (double*)malloc(rows * cols * sizeof(double));
}

/**
 * Inicialización con valores aleatorios
 */
void init_matrix(double* m, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        m[i] = (double)(rand() % 10);
    }
}

/**
 * Multiplicación de matrices optimizada para OpenACC
 */
void multiply_acc(double* A, double* B, double* C, int M, int K, int N) {
    // Gestionamos la transferencia de datos a la GPU
    #pragma acc data copyin(A[0:M*K], B[0:K*N]) copyout(C[0:M*N])
    {
        // 'parallel loop' asegura el paralelismo
        // 'collapse(2)' fusiona los bucles i y j para maximizar el uso de la GPU
        #pragma acc parallel loop collapse(2)
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                double sum = 0.0;
                for (int k = 0; k < K; k++) {
                    sum += A[i * K + k] * B[k * N + j];
                }
                C[i * N + j] = sum;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Uso: %s <filas_A> <cols_A_filas_B> <cols_B>\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);

    if (M <= 0 || K <= 0 || N <= 0) {
        printf("Error: Las dimensiones deben ser mayores que 0\n");
        return 1;
    }

    srand(time(NULL));

    // Reserva de memoria en el Host (CPU)
    double* A = allocate_matrix_flat(M, K);
    double* B = allocate_matrix_flat(K, N);
    double* C = allocate_matrix_flat(M, N);

    if (A == NULL || B == NULL || C == NULL) {
        printf("Error: No se pudo asignar memoria.\n");
        return 1;
    }

    init_matrix(A, M, K);
    init_matrix(B, K, N);

    struct timespec start_ts, end_ts;
    
    // Medición de tiempo
    clock_gettime(CLOCK_MONOTONIC, &start_ts);

    multiply_acc(A, B, C, M, K, N);

    clock_gettime(CLOCK_MONOTONIC, &end_ts);
    
    double elapsed = (end_ts.tv_sec - start_ts.tv_sec) + 
                     (end_ts.tv_nsec - start_ts.tv_nsec) / 1e9;

    printf("--- Resultado OpenACC ---\n");
    printf("Dimensiones: A(%dx%d) * B(%dx%d) -> C(%dx%d)\n", M, K, K, N, M, N);
    printf("Tiempo de ejecucion: %.6f segundos\n", elapsed);

    // Liberación de memoria
    free(A);
    free(B);
    free(C);

    return 0;
}