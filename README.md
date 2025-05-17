# Programación paralela de la multiplicación de matrices

Este proyecto evalúa el rendimiento de la multiplicación de matrices cuadradas usando cuatro tecnologías diferentes:
**Base (secuencial), OpenMP, CUDA y OpenMPI**. Se incluyen todos los materiales requeridos para su análisis y presentación.

---

## Estructura del Proyecto

```
├── Base.c              # Código base secuencial
├── OpenMP.c            # Código con paralelización usando OpenMP
├── Cuda.cu             # Código ejecutado en GPU con CUDA
├── OpenMPI.c           # Código base OpenMPI
│  
├── ejecutor.sh         # Script para ejecutar todos los códigos automáticamente              
├── resultado_Base.txt
├── resultado_OpenMP.txt
├── resultado_CUDA.txt
├── resultado_OpenMPI.txt

├── ejecucion.mp4             # Video demostrando cómo se ejecutan los cuatro códigos
├── resultados.xlsx           # Archivo Excel con todos los datos estructurados y graficados

├── memoria.pdf               # Informe completo del proyecto
└── README.md                 # Este archivo
```

---

## Compilación

### Base (Secuencial)

```bash
gcc -o Base Base.c -lpapi
```

### OpenMP

```bash
gcc -o OpenMP OpenMP.c -fopenmp -lpapi
```

### CUDA

```bash
nvcc -o Cuda Cuda.cu -lpapi
```

### OpenMPI

```bash
mpicc -o OpenMPI OpenMPI.c -lpapi
```

Se puede ejecutar todos los codigos con con:

```bash
./ejecutor.sh
```