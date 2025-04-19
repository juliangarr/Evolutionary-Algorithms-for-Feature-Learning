# Proyecto IC

## Estructura de Directorios del Proyecto

```plaintext
BIN/                             # Carpeta que contiene datos y scripts relacionados con la ejecución
├── datasets_arff/               # Conjunto de datos en formato ARFF utilizados para entrenar y evaluar los modelos
├── results/                     # Resultados generados por los experimentos (CSVs con las métricas)
├── main                         # Ejecutable principal del proyecto
└── run.sh                       # Script de ejecución automática del proyecto

FUENTES/                         # Contiene el código fuente y archivos de compilación
├── build/                       # Archivos generados durante la compilación (binarios, objetos, etc.)
├── include/                     # Archivos de encabezado (.h), con definiciones de clases y funciones
├── src/                         # Código fuente principal del proyecto (.cc)
└── Makefile                     # Archivo Makefile para compilar el proyecto
```

## Estructura del código de los algoritmos evolutivos

En mi código implemento una estructura de herencia de clases para no repetir código.

Para los algoritmos genéticos, he definido la clase GeneticClassifier para encapsular el esquema de evolución común tanto del AGG como del AGE en un método llamado:
    - AG. 
    
Además, en GeneticClassifier implemento los métodos siguientes que también son comunes a ambos (AGG y AGE):
    - evaluarPoblacion
    - mejorIndividuo
    - generarPadres
    - seleccionTorneo
    - cruce
    - cruceBLX
    - cruceAritmetico

Luego, defino las clases AGGClassifier y AGEClassifier que heredan de GeneticClassifier e implementan solamente los métodos que difieren en ambos esquemas:
    - mutacion
    - reemplazo

Por lo tanto, para implementar un clasificador AGG o AGE, en lugar de llamar a AGG/AGE, llamo a AG y esta se encarga de llamar a las funciones de mutacion/reemplazo correspondientes según la clase del objeto clasificador creado.

Para los algoritmos meméticos lo hago de manera análoga: He definido la clase AMProbClassifier y el método AM_Prob para encapsular tanto el calasificador AM-ALL como el AM-RAND en un clasificador padre donde varía la cantidad de población escogida para la BL según una probabilidad como parámetro. Si la probabilidad de AMProbClassifier es 1.0, entonces el clasificador resultante será un AM-ALL y si la probabilidad de parámetro es 0.1, entonces el clasificador resultante será AM-RAND.