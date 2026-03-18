#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

// Reservar memoria
double** allocate_matrix(int n) {
    double** m = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        m[i] = (double*)malloc(n * sizeof(double));
    }
    return m;
}


// Liberar memoria
void free_matrix(double** m, int n) {
    for (int i = 0; i < n; i++) {
        free(m[i]);
    }
    free(m);
}

// Inicializar matriz
void init_matrix(double** m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i][j] = rand() % 10;
        }
    }
}

// Multiplicación paralela con OpenMP
void multiply_omp(double** A, double** B, double** C, int n) {

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

int main(int argc, char* argv[]) {

    if (argc != 2) {
        printf("Uso: %s <tamano_matriz>\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    if (n <= 0) {
        printf("El tamaño debe ser mayor que 0\n");
        return 1;
    }

    srand(time(NULL));

    double** A = allocate_matrix(n);
    double** B = allocate_matrix(n);
    double** C = allocate_matrix(n);

    init_matrix(A, n);
    init_matrix(B, n);

    double start = omp_get_wtime();

    multiply_omp(A, B, C, n);

    double end = omp_get_wtime();

    printf("Tamaño: %dx%d\n", n, n);
    printf("Hilos usados: %d\n", omp_get_max_threads());
    printf("Tiempo de ejecucion: %.4f segundos\n", end - start);

    // Checksum (opcional, pero recomendable)
    double checksum = 0.0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            checksum += C[i][j];

    printf("Checksum: %.2f\n", checksum);

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}