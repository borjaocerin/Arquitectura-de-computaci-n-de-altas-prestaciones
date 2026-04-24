#!/bin/bash

# Nombres de los ejecutables
EXE_SEQ="./secuencial"
EXE_OMP="./practicaomp"
EXE_MPI="./practicampi"
EXE_ACC="./practicaacc"

# Configuración de las pruebas
# Tamaños de matriz (M K N). Usamos múltiplos de 16 para que MPI no falle.
SIZES=("512 512 512" "1024 1024 1024" "2048 2048 2048")
CORES=(1 2 4 8 16)

# Archivo de resultados
OUTPUT="resultados_rendimiento.csv"
echo "Algoritmo,Dimensiones,Nucleos,Tiempo(s)" > $OUTPUT

# 1. Compilación
echo "Compilando programas..."
gcc secuencial.c -o secuencial
gcc -fopenmp practicaomp.c -o practicaomp
mpicc practicampi.c -o practicampi
nvc -acc -Minfo=accel practicaacc.c -o practicaacc

echo "Iniciando pruebas de rendimiento..."

for SIZE in "${SIZES[@]}"; do
    M=$(echo $SIZE | cut -d' ' -f1)
    K=$(echo $SIZE | cut -d' ' -f2)
    N=$(echo $SIZE | cut -d' ' -f3)
    
    echo "--- Probando tamaño $M x $N ---"

    # --- EJECUCIÓN SECUENCIAL ---
    # Solo se corre una vez por tamaño ya que no escala con núcleos
    TIME_SEQ=$($EXE_SEQ $M $K $N | grep "Tiempo" | awk '{print $4}')
    echo "Secuencial,$M x $N,1,$TIME_SEQ" >> $OUTPUT

    for P in "${CORES[@]}"; do
        echo "  Ejecutando con $P núcleos..."

        # --- EJECUCIÓN OPENMP ---
        export OMP_NUM_THREADS=$P
        TIME_OMP=$($EXE_OMP $M $K $N | grep "Tiempo" | awk '{print $4}')
        echo "OpenMP,$M x $N,$P,$TIME_OMP" >> $OUTPUT

        # --- EJECUCIÓN MPI ---
        TIME_MPI=$(mpirun -np $P --use-hwthread-cpus $EXE_MPI $M $K $N | grep "Tiempo" | awk '{print $4}')
        echo "MPI,$M x $N,$P,$TIME_MPI" >> $OUTPUT
        
        # --- EJECUCIÓN OPENACC ---
        # Si se ejecuta en multicore CPU, responde a hilos. Si es GPU, el tiempo será constante.
        TIME_ACC=$($EXE_ACC $M $K $N | grep "Tiempo" | awk '{print $4}')
        echo "OpenACC,$M x $N,$P,$TIME_ACC" >> $OUTPUT
    done
done

echo "Estudio finalizado. Los datos están en $OUTPUT"