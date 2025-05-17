#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <papi.h>        

// Estructura que almacena los resultados del experimento
typedef struct {
    long long instrucciones;   // Número total de instrucciones ejecutadas
    long long ciclos;          // Número total de ciclos de CPU
    double tiempo;             // Tiempo promedio por ejecución
    double tiempo_total;       // Tiempo total (en caso de múltiples repeticiones)
    long long C;               // Suma de todos los elementos de la matriz resultante
} Resultados;

// Kernel de CUDA: cada hilo calculará un elemento de la matriz resultante C
__global__ void multiplicarKernel(int* A, int* B, long long* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // Índice global de fila
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // Índice global de columna

    // Solo si está dentro de los límites de la matriz
    if (row < N && col < N) {
        long long sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += (long long) A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

void calcularEnCUDA(int* A, int* B, long long* C, int N) {
    int *d_A, *d_B;
    long long *d_C;

    // Reservar memoria en la GPU para A, B y C
    cudaMalloc((void**)&d_A, N * N * sizeof(int));
    cudaMalloc((void**)&d_B, N * N * sizeof(int));
    cudaMalloc((void**)&d_C, N * N * sizeof(long long));

    // Copiar matrices A y B desde la CPU a la GPU
    cudaMemcpy(d_A, A, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(int), cudaMemcpyHostToDevice);

    // Definición del tamaño de bloque e invocación del grid
    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (N + 15) / 16);

    // Lanzar el kernel CUDA
    multiplicarKernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();  // Esperar a que todos los hilos terminen

    // Copiar el resultado desde la GPU a la CPU
    cudaMemcpy(C, d_C, N * N * sizeof(long long), cudaMemcpyDeviceToHost);

    // Liberar la memoria de la GPU
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

// Función que realiza una única multiplicación de matrices usando CUDA
Resultados simularMultiplicacionMatrices(int N) {
    Resultados resultado;
    int eventSet = PAPI_NULL;
    long long valores[2];
    resultado.C = 0;

    // Reservar de memoria 
    int* A = (int*)malloc(N * N * sizeof(int));
    int* B = (int*)malloc(N * N * sizeof(int));
    long long* C = (long long*)malloc(N * N * sizeof(long long));

    if (!A || !B || !C) {
        fprintf(stderr, "Error al reservar memoria para las matrices\n");
        exit(1);
    }

    // Inicializar matrices A y B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
            B[i * N + j] = 2 * N - 2 - i - j;
            C[i * N + j] = 0;
        }
    }

    // Configurar PAPI
    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&eventSet);
    PAPI_add_event(eventSet, PAPI_TOT_INS);   // Medir instrucciones
    PAPI_add_event(eventSet, PAPI_TOT_CYC);   // Medir ciclos

    long long start_time = PAPI_get_real_usec();
    PAPI_reset(eventSet);
    PAPI_start(eventSet);

    // // Multiplicación real en la GPU
    calcularEnCUDA(A, B, C, N);

    // Sumar todos los valores de C
    for (int i = 0; i < N * N; i++) {
        resultado.C += C[i];
    }

    PAPI_stop(eventSet, valores);
    long long end_time = PAPI_get_real_usec();

    resultado.instrucciones = valores[0];
    resultado.ciclos = valores[1];
    resultado.tiempo = (end_time - start_time) / 1e6;
    resultado.tiempo_total = resultado.tiempo;

    // Liberar memoria de CPU
    free(A);
    free(B);
    free(C);

    // Finalizar PAPI
    PAPI_cleanup_eventset(eventSet);
    PAPI_destroy_eventset(&eventSet);
    PAPI_shutdown();

    return resultado;
}

Resultados simularMultiplicacionConRepeticiones(int N, int repeticiones) {
    Resultados resultado;
    int eventSet = PAPI_NULL;
    long long valores[2];
    long long C_total = 0;
    long long totalInstrucciones = 0;
    long long totalCiclos = 0;
    double totalTiempo = 0.0;

    // Reservar memoria en la CPU
    int* A = (int*)malloc(N * N * sizeof(int));
    int* B = (int*)malloc(N * N * sizeof(int));
    long long* C = (long long*)malloc(N * N * sizeof(long long));

    if (!A || !B || !C) {
        fprintf(stderr, "Error al reservar memoria\n");
        exit(1);
    }

    // Inicializar matrices A y B
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = i + j;
            B[i * N + j] = 2 * N - 2 - i - j;
        }
    }

    // Configurar PAPI
    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&eventSet);
    PAPI_add_event(eventSet, PAPI_TOT_INS);
    PAPI_add_event(eventSet, PAPI_TOT_CYC);

    // Repetir el cálculo tantas veces como se indique
    for (int r = 0; r < repeticiones; r++) {
        memset(C, 0, N * N * sizeof(long long));  // Reiniciar C

        long long start_time = PAPI_get_real_usec();
        PAPI_reset(eventSet);
        PAPI_start(eventSet);

        calcularEnCUDA(A, B, C, N);  // Ejecutar cálculo en CUDA

        long long sumaC = 0;
        for (int i = 0; i < N * N; i++) {
            sumaC += C[i];
        }

        PAPI_stop(eventSet, valores);
        long long end_time = PAPI_get_real_usec();

        totalInstrucciones += valores[0];
        totalCiclos += valores[1];
        totalTiempo += (end_time - start_time) / 1e6;
        C_total = sumaC;  // Última suma válida

        // Imprimir progreso (cada 10% del total)
        if (repeticiones >= 10 && r % (repeticiones / 10) == 0) {
            printf("Iteración %d de %d (%.0f%%)\n", r + 1, repeticiones, (100.0 * (r + 1)) / repeticiones);
        }
    }

    // Liberar memoria
    free(A);
    free(B);
    free(C);

    PAPI_cleanup_eventset(eventSet);
    PAPI_destroy_eventset(&eventSet);
    PAPI_shutdown();

    // Calcular promedios y retornar
    resultado.instrucciones = totalInstrucciones / repeticiones;
    resultado.ciclos = totalCiclos / repeticiones;
    resultado.tiempo = totalTiempo / repeticiones;
    resultado.tiempo_total = totalTiempo;
    resultado.C = C_total;

    return resultado;
}

// Función principal
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
    printf("  0 - Ejecución normal (CUDA)\n  1 - Ejecución múltiple (CUDA)\n");
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

    int tamanos[8] = {3, 10, 100, 250, 500, 1000, 1500, 2000};
    int n = tamanos[caso_matriz];

    Resultados res;
    if (tipo_ejecucion == 0) {
        res = simularMultiplicacionMatrices(n);
    } else {
        res = simularMultiplicacionConRepeticiones(n, extra);
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