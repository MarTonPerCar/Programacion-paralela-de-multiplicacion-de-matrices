#include <stdio.h>
#include <stdlib.h>
#include <papi.h>
#include <string.h>
#include <omp.h>

// Estructura que almacena los resultados del experimento
typedef struct {
    long long instrucciones;   // Número total de instrucciones ejecutadas
    long long ciclos;          // Número total de ciclos de CPU
    double tiempo;             // Tiempo promedio por ejecución
    double tiempo_total;       // Tiempo total (en caso de múltiples repeticiones)
    long long C;               // Suma de todos los elementos de la matriz resultante
} Resultados;

// Simulación de una sola ejecución con OpenMP
Resultados simularMultiplicacionMatricesOpenMP(int N) {
    Resultados resultado;
    int eventSet = PAPI_NULL;
    long long valores[2];
    resultado.C = 0;

    int* A = malloc(N * N * sizeof(int));
    int* B = malloc(N * N * sizeof(int));
    long long* C = malloc(N * N * sizeof(long long));

    if (!A || !B || !C) {
        fprintf(stderr, "Error al reservar memoria\n");
        exit(1);
    }

    // Inicializar A y B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
            B[i * N + j] = 2 * N - 2 - i - j;
            C[i * N + j] = 0;
        }
    }

    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&eventSet);
    PAPI_add_event(eventSet, PAPI_TOT_INS);
    PAPI_add_event(eventSet, PAPI_TOT_CYC);

    long long start_time = PAPI_get_real_usec();
    PAPI_reset(eventSet);
    PAPI_start(eventSet);

    // Multiplicación con OpenMP
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C[i * N + j] += (long long) A[i * N + k] * B[k * N + j];
            }
        }
    }

    // Suma total de C
    long long sumaC = 0;
    #pragma omp parallel for reduction(+:sumaC)
    for (int i = 0; i < N * N; i++) {
        sumaC += C[i];
    }

    PAPI_stop(eventSet, valores);
    long long end_time = PAPI_get_real_usec();

    resultado.instrucciones = valores[0];
    resultado.ciclos = valores[1];
    resultado.tiempo = (end_time - start_time) / 1e6;
    resultado.tiempo_total = resultado.tiempo;
    resultado.C = sumaC;

    free(A);
    free(B);
    free(C);

    PAPI_cleanup_eventset(eventSet);
    PAPI_destroy_eventset(&eventSet);
    PAPI_shutdown();

    return resultado;
}

// Ejecuciones múltiples con OpenMP
Resultados simularMultiplicacionConRepeticionesOpenMP(int N, int repeticiones) {
    Resultados resultado;
    int eventSet = PAPI_NULL;
    long long valores[2];

    long long totalInstrucciones = 0;
    long long totalCiclos = 0;
    double totalTiempo = 0.0;
    long long C_acumulado = 0;

    int* A = malloc(N * N * sizeof(int));
    int* B = malloc(N * N * sizeof(int));
    long long* C = malloc(N * N * sizeof(long long));

    if (!A || !B || !C) {
        fprintf(stderr, "Error al reservar memoria\n");
        exit(1);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
            B[i * N + j] = 2 * N - 2 - i - j;
        }
    }

    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&eventSet);
    PAPI_add_event(eventSet, PAPI_TOT_INS);
    PAPI_add_event(eventSet, PAPI_TOT_CYC);

    for (int r = 0; r < repeticiones; r++) {
        if (repeticiones >= 10 && r % (repeticiones / 10) == 0) {
            printf("Iteración %d de %d (%.0f%%)\n", r + 1, repeticiones, (100.0 * (r + 1)) / repeticiones);
            fflush(stdout);
        }

        memset(C, 0, N * N * sizeof(long long));

        long long start_time = PAPI_get_real_usec();
        PAPI_reset(eventSet);
        PAPI_start(eventSet);

        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    C[i * N + j] += (long long) A[i * N + k] * B[k * N + j];
                }
            }
        }

        long long sumaC = 0;
        #pragma omp parallel for reduction(+:sumaC)
        for (int i = 0; i < N * N; i++) {
            sumaC += C[i];
        }

        PAPI_stop(eventSet, valores);
        long long end_time = PAPI_get_real_usec();

        totalInstrucciones += valores[0];
        totalCiclos += valores[1];
        totalTiempo += (end_time - start_time) / 1e6;
        C_acumulado = sumaC;
    }

    free(A);
    free(B);
    free(C);

    PAPI_cleanup_eventset(eventSet);
    PAPI_destroy_eventset(&eventSet);
    PAPI_shutdown();

    resultado.instrucciones = totalInstrucciones / repeticiones;
    resultado.ciclos = totalCiclos / repeticiones;
    resultado.tiempo = totalTiempo / repeticiones;
    resultado.tiempo_total = totalTiempo;
    resultado.C = C_acumulado;

    return resultado;
}


int main() {
    int caso_matriz, tipo_ejecucion, extra = 0;

    printf("Selecciona el caso de matriz (0 a 7):\n");
    printf("  0 - Matriz 3x3\n  1 - Matriz 10x10\n  2 - Matriz 100x100\n");
    printf("  3 - Matriz 250x250\n  4 - Matriz 500x500\n");
    printf("  5 - Matriz 1,000x1,000\n  6 - Matriz 1,500x1,500\n");
    printf("  7 - Matriz 2,000x2,000\n");
    printf("Ingresa tu opción: ");
    scanf("%d", &caso_matriz);
    if (caso_matriz < 0 || caso_matriz > 7) {
        printf("Error: caso de matriz inválido.\n");
        return 1;
    }

    printf("\nSelecciona el tipo de ejecución (0 o 1):\n");
    printf("  0 - Ejecución normal\n  1 - Ejecución múltiple\n");
    scanf("%d", &tipo_ejecucion);
    if (tipo_ejecucion < 0 || tipo_ejecucion > 1) {
        printf("Error: tipo de ejecución inválido.\n");
        return 1;
    }

    if (tipo_ejecucion == 1) {
        printf("Ingresa un número extra para ejecución múltiple: ");
        scanf("%d", &extra);
        if (extra <= 0) {
            printf("Error: número de repeticiones inválido.\n");
            return 1;
        }
    }

    int tamaños[8] = {3, 10, 100, 250, 500, 1000, 1500, 2000};
    int n = tamaños[caso_matriz];

    Resultados res;
    if (tipo_ejecucion == 0) {
        res = simularMultiplicacionMatricesOpenMP(n);
    } else {
        res = simularMultiplicacionConRepeticionesOpenMP(n, extra);
    }

    printf("\n=== RESULTADOS ===\n");
    printf("Tamaño de matriz: %dx%d\n", n, n);
    printf("Instrucciones promedio: %lld\n", res.instrucciones);
    printf("Ciclos promedio: %lld\n", res.ciclos);
    printf("Tiempo promedio: %.6f segundos\n", res.tiempo);
    if (tipo_ejecucion == 1) {
    	printf("Tiempo total: %.6f segundos\n", res.tiempo_total);
    }
    printf("Valor acumulado C: %lld\n", res.C);

    return 0;
}