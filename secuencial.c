#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Modificada para aceptar filas y columnas
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

// Algoritmo adaptado a dimensiones M, K, N
void multiply(double** A, double** B, double** C, int M, int K, int N) {
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
    // Ahora pedimos 3 parámetros
    if (argc != 4) {
        printf("Uso: %s <filas_A> <cols_A_filas_B> <cols_B>\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]); // Filas A
    int K = atoi(argv[2]); // Columnas A y Filas B
    int N = atoi(argv[3]); // Columnas B

    if (M <= 0 || K <= 0 || N <= 0) {
        printf("Las dimensiones deben ser mayores que 0\n");
        return 1;
    }

    srand(time(NULL));

    // Reservar matrices con sus dimensiones correspondientes
    double** A = allocate_matrix(M, K);
    double** B = allocate_matrix(K, N);
    double** C = allocate_matrix(M, N); // El resultado es de M x N

    init_matrix(A, M, K);
    init_matrix(B, K, N);

    clock_t start = clock();

    multiply(A, B, C, M, K, N);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Dimensiones: A(%dx%d) * B(%dx%d) -> C(%dx%d)\n", M, K, K, N, M, N);
    printf("Tiempo de ejecucion: %.4f segundos\n", time_spent);

    free_matrix(A, M);
    free_matrix(B, K);
    free_matrix(C, M);

    return 0;
}