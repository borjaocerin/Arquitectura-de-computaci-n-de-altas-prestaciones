#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Función para reservar memoria para una matriz
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

// Inicializar matriz con valores aleatorios
void init_matrix(double** m, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            m[i][j] = rand() % 10; // valores 0–9
        }
    }
}

// Multiplicación de matrices (secuencial)
void multiply(double** A, double** B, double** C, int n) {
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

    // Reservar matrices
    double** A = allocate_matrix(n);
    double** B = allocate_matrix(n);
    double** C = allocate_matrix(n);

    // Inicializar
    init_matrix(A, n);
    init_matrix(B, n);

    // Medir tiempo
    clock_t start = clock();

    multiply(A, B, C, n);

    clock_t end = clock();
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Tamaño: %dx%d\n", n, n);
    printf("Tiempo de ejecucion: %.4f segundos\n", time_spent);

    // Liberar memoria
    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C, n);

    return 0;
}