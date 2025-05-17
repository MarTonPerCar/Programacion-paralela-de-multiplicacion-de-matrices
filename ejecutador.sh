#!/bin/bash

# Nombres de los ejecutables
EJECUTABLES=("Codigos_Base/MatrizTest" "Codigos_OpenMP/Codigos_MOMP" "Codigos_Cuda/CUDA" "Codigos_OpenMPI/OMPI")
NOMBRES=("Ejecución Normal" "Ejecución OpenMP" "Ejecución CUDA" "Ejecución OpenMPI")
REPETICIONES=100
TIPOS=("3x3" "10x10" "100x100" "250x250" "500x500" "1000x1000" "1500x1500" "2000x2000")

for idx in "${!EJECUTABLES[@]}"; do
    exe="${EJECUTABLES[$idx]}"
    nombre="${NOMBRES[$idx]}"
    archivo_resultado="resultado_${idx}.txt"

    echo "Guardando resultados en $archivo_resultado"
    > "$archivo_resultado"

    echo ">>> $nombre - Modo normal <<<" >> "$archivo_resultado"
    for i in {0..7}; do
        tipo=${TIPOS[$i]}
        echo "Caso $i - Ejecución normal (0)" >> "$archivo_resultado"

        if [[ "$nombre" == "Ejecución OpenMPI" ]]; then
            mpirun -np 4 "./$exe" $i 0 | sed -n '/=== RESULTADOS/,${p}' >> "$archivo_resultado"
        else
            echo -e "$i\n0" | ./"$exe" | sed -n '/=== RESULTADOS/,${p}' >> "$archivo_resultado"
        fi

        echo "" >> "$archivo_resultado"
        echo "Finalizado la matriz de tipo $tipo con caso 0 de $nombre"
    done

    echo ">>> $nombre - Modo múltiple ($REPETICIONES repeticiones) <<<" >> "$archivo_resultado"
    for i in {0..7}; do
        tipo=${TIPOS[$i]}
        echo "Caso $i - Ejecución múltiple (1), repeticiones = $REPETICIONES" >> "$archivo_resultado"

        if [[ "$nombre" == "Ejecución OpenMPI" ]]; then
            mpirun -np 4 "./$exe" $i 1 $REPETICIONES | sed -n '/=== RESULTADOS/,${p}' >> "$archivo_resultado"
        else
            echo -e "$i\n1\n$REPETICIONES" | ./"$exe" | sed -n '/=== RESULTADOS/,${p}' >> "$archivo_resultado"
        fi

        echo "" >> "$archivo_resultado"
        echo "Finalizado la matriz de tipo $tipo con caso 1 de $nombre"
    done

    echo "Finalizado: $exe"
    echo ""
done
