#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <papi.h>

// Estructura para los resultados
typedef struct {
    long long instrucciones;
    long long ciclos;
    double tiempo;
    double tiempo_total;
    long long C;
} Resultados;

Resultados simularMultiplicacionMatricesMPI(int N) {
    Resultados resultado = {0};
    int eventSet = PAPI_NULL;
    long long valores[2] = {0};
    resultado.C = 0;

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int filas_local = N / size;
    int resto = N % size;
    if (rank < resto) filas_local++;

    int offset = (N / size) * rank + (rank < resto ? rank : resto);

    int* A_local = malloc(filas_local * N * sizeof(int));
    int* B = malloc(N * N * sizeof(int));
    long long* C_local = calloc(filas_local * N, sizeof(long long));

    if (!A_local || !B || !C_local) {
        fprintf(stderr, "Error al reservar memoria\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int* A_full = NULL;
    if (rank == 0) {
        A_full = malloc(N * N * sizeof(int));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A_full[i * N + j] = i + j;
                B[i * N + j] = 2 * N - 2 - i - j;
            }
        }
    }

    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    int* sendcounts = malloc(size * sizeof(int));
    int* displs = malloc(size * sizeof(int));
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (N / size + (i < resto ? 1 : 0)) * N;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    MPI_Scatterv(A_full, sendcounts, displs, MPI_INT, A_local, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) free(A_full);

    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&eventSet);
    PAPI_add_event(eventSet, PAPI_TOT_INS);
    PAPI_add_event(eventSet, PAPI_TOT_CYC);

    long long start_time = PAPI_get_real_usec();
    PAPI_reset(eventSet);
    PAPI_start(eventSet);

    for (int i = 0; i < filas_local; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                C_local[i * N + j] += (long long) A_local[i * N + k] * B[k * N + j];
            }
        }
    }

    long long suma_local = 0;
    for (int i = 0; i < filas_local * N; i++) {
        suma_local += C_local[i];
    }

    PAPI_stop(eventSet, valores);
    long long end_time = PAPI_get_real_usec();

    long long suma_total = 0;
    MPI_Reduce(&suma_local, &suma_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        resultado.C = suma_total;
        resultado.instrucciones = valores[0];
        resultado.ciclos = valores[1];
        resultado.tiempo = (end_time - start_time) / 1e6;
        resultado.tiempo_total = resultado.tiempo;
    }

    free(A_local);
    free(B);
    free(C_local);
    free(sendcounts);
    free(displs);

    PAPI_cleanup_eventset(eventSet);
    PAPI_destroy_eventset(&eventSet);
    PAPI_shutdown();

    return resultado;
}

Resultados simularMultiplicacionConRepeticionesMPI(int N, int repeticiones) {
    Resultados resultado = {0};
    int eventSet = PAPI_NULL;
    long long valores[2];

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int filas_local = N / size;
    int resto = N % size;
    if (rank < resto) filas_local++;

    int* A_local = malloc(filas_local * N * sizeof(int));
    int* B = malloc(N * N * sizeof(int));
    long long* C_local = malloc(filas_local * N * sizeof(long long));

    if (!A_local || !B || !C_local) {
        fprintf(stderr, "Error al reservar memoria\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int* A_full = NULL;
    if (rank == 0) {
        A_full = malloc(N * N * sizeof(int));
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                A_full[i * N + j] = i + j;
                B[i * N + j] = 2 * N - 2 - i - j;
            }
        }
    }

    MPI_Bcast(B, N * N, MPI_INT, 0, MPI_COMM_WORLD);

    int* sendcounts = malloc(size * sizeof(int));
    int* displs = malloc(size * sizeof(int));
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sendcounts[i] = (N / size + (i < resto ? 1 : 0)) * N;
        displs[i] = sum;
        sum += sendcounts[i];
    }

    MPI_Scatterv(A_full, sendcounts, displs, MPI_INT, A_local, sendcounts[rank], MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) free(A_full);

    PAPI_library_init(PAPI_VER_CURRENT);
    PAPI_create_eventset(&eventSet);
    PAPI_add_event(eventSet, PAPI_TOT_INS);
    PAPI_add_event(eventSet, PAPI_TOT_CYC);

    long long totalInstrucciones = 0, totalCiclos = 0;
    double totalTiempo = 0.0;
    long long C_acumulado = 0;

    for (int r = 0; r < repeticiones; r++) {
        memset(C_local, 0, filas_local * N * sizeof(long long));

        long long start_time = PAPI_get_real_usec();
        PAPI_reset(eventSet);
        PAPI_start(eventSet);

        for (int i = 0; i < filas_local; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    C_local[i * N + j] += (long long) A_local[i * N + k] * B[k * N + j];
                }
            }
        }

        long long suma_local = 0;
        for (int i = 0; i < filas_local * N; i++) {
            suma_local += C_local[i];
        }

        PAPI_stop(eventSet, valores);
        long long end_time = PAPI_get_real_usec();

        totalInstrucciones += valores[0];
        totalCiclos += valores[1];
        totalTiempo += (end_time - start_time) / 1e6;

        if (rank == 0) C_acumulado = suma_local;
    }

    long long suma_total = 0;
    MPI_Reduce(&C_acumulado, &suma_total, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        resultado.C = suma_total;
        resultado.instrucciones = totalInstrucciones / repeticiones;
        resultado.ciclos = totalCiclos / repeticiones;
        resultado.tiempo = totalTiempo / repeticiones;
        resultado.tiempo_total = totalTiempo;
    }

    free(A_local);
    free(B);
    free(C_local);
    free(sendcounts);
    free(displs);

    PAPI_cleanup_eventset(eventSet);
    PAPI_destroy_eventset(&eventSet);
    PAPI_shutdown();

    return resultado;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int caso_matriz = 0, tipo_ejecucion = 0, repeticiones = 0;
    if (argc < 3) {
        printf("Uso: %s <caso matriz 0..7> <tipo_ejecucion 0|1> [repeticiones]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    sscanf(argv[1], "%d", &caso_matriz);
    sscanf(argv[2], "%d", &tipo_ejecucion);
    if (tipo_ejecucion == 1 && argc >= 4) sscanf(argv[3], "%d", &repeticiones);

    int tamaños[8] = {3, 10, 100, 250, 500, 1000, 1500, 2000};
    int N = tamaños[caso_matriz];

    Resultados res;
    if (tipo_ejecucion == 0) {
        res = simularMultiplicacionMatricesMPI(N);
    } else {
        res = simularMultiplicacionConRepeticionesMPI(N, repeticiones);
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        printf("\n=== RESULTADOS (MPI) ===\n");
        printf("Tamaño de matriz: %dx%d\n", N, N);
        printf("Instrucciones promedio: %lld\n", res.instrucciones);
        printf("Ciclos promedio: %lld\n", res.ciclos);
        printf("Tiempo promedio: %.6f segundos\n", res.tiempo);
        if (tipo_ejecucion == 1)
            printf("Tiempo total: %.6f segundos\n", res.tiempo_total);
        printf("Valor acumulado C: %lld\n", res.C);
    }

    MPI_Finalize();
    return 0;
}