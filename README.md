# persona-cluster

## Descripción

**persona-cluster** es un programa escrito en Python que permite identificar personas a partir de las coletillas que
usan al hablar y escribir. Utilizando conceptos del criptoanálisis y técnicas de machine learning, el programa analiza
la frecuencia de palabras en el texto, crea modelos, los guarda y los usa para inferir la identidad de los hablantes.
El programa incluye un sistema de registro para seguir los pasos del proceso de manera didáctica.

## Requisitos

- Python 3.7 o superior
- Bibliotecas:
    - `numpy`
    - `scipy`
    - `collections`
    - `pickle`
    - `logging`

## Estructura del Proyecto

```
.
├── data
│   ├── models
│   │   └── cuatro_personas-thr_7.pkl
│   └── train
│       ├── Elías.txt
│       ├── Marta.txt
│       ├── Pedro.txt
│       └── Raúl.txt
├── src
│   ├── main.py
│   ├── text_clustering.py
│   ├── train.py
│   └── utils.py
└── text_clustering.log
```

## Instalación

1. Clonar el repositorio:
    ```sh
    git clone https://github.com/gorkapuenteusal/persona-cluster.git
    cd persona-cluster
    ```

2. Instalar las dependencias:
    ```sh
    pip install numpy scipy
    ```

## Uso

### Inicialización del Modelo

```python
from text_clustering import TextClustering
import logging

logging.basicConfig(level=logging.INFO)

model = TextClustering(threshold=0.5)
```

### Añadir Texto para Entrenamiento

```python
model.add_text("O sea, era un espectáculo, o sea, todo el mundo se quedó mirando. O sea, me acerqué al dueño y le pregunté, o sea, cómo había entrenado al perro así.", name="Marta")
model.add_text("Eeeh, hoy fue un día bastante interesante, eeeh, porque fui a un museo de arte moderno.", name="Pedro")
model.add_text("Hola, soy Raúl, mmmh. Hoy he decidido salir a correr por la mañana, mmmh, porque el clima está perfecto, mmmh.", name="Raúl")
model.add_text("Primero, opa, fui al gimnasio para una sesión de entrenamiento, opa. Luego, opa, decidí probar una nueva receta de pasta para el almuerzo, opa.", name="Elías")
```

### Guardar el Modelo

```python
model.save_model("persona_cluster_model.pkl")
```

### Cargar el Modelo

```python
loaded_model = TextClustering.load_model("persona_cluster_model.pkl")
```

### Inferir la Identidad de un Nuevo Texto

```python
result = loaded_model.predict("O sea, es increíble cómo sucedió todo, o sea, no me lo esperaba.")
print(f"Texto identificado como: {result}")
```

### Ejecución desde la Línea de Comandos

```sh
python src/main.py --train Marta.txt Pedro.txt Raúl.txt Elías.txt --threshold 7
python src/main.py --model cuatro_personas-thr_7 --threshold 9
```

## Ejemplo Práctico

Se han creado cuatro personas virtuales utilizando las siguientes coletillas:

- Marta: "O sea"
- Pedro: "Eeeh"
- Raúl: "Mmmh"
- Elías: "Opa"

Cada persona generará frases cortas usando frecuentemente su coletilla específica. Se han utilizado 20-25 frases de
tamaño similar para cada persona en el entrenamiento del modelo con un threshold de 7. Para la inferencia, se usaron 4
frases de cada uno y un threshold de 9. Los resultados obtenidos fueron:

| Persona Real \ Identificada como | Marta | Pedro | Raúl | Elías |
|----------------------------------|-------|-------|------|-------|
| **Marta**                        | 4     | 0     | 0    | 0     |
| **Pedro**                        | 0     | 3     | 1    | 0     |
| **Raúl**                         | 0     | 0     | 4    | 0     |
| **Elías**                        | 0     | 0     | 0    | 4     |

- **% de acierto**: 93.75%

## Conclusión

El programa "persona-cluster" demuestra alta precisión al identificar personas por sus coletillas específicas. Sin
embargo, hubo un error de identificación en el caso de Pedro, indicando que el modelo puede mejorar en la discriminación
de coletillas con características contextuales similares.

## Funciones Principales

### main.py

- **main()**: Gestiona la línea de comandos para entrenar o cargar modelos y realizar predicciones.

### text_clustering.py

- **preprocess_text(text)**: Convierte el texto a minúsculas y lo divide en palabras.
- **calculate_ngrams(words, n)**: Calcula los n-gramas de una lista de palabras.
- **calculate_frequencies(words)**: Calcula las frecuencias de unigrams, bigrams y trigrams.
- **vectorize_frequencies(frequencies)**: Convierte las frecuencias en un vector basado en el vocabulario conocido.
- **calculate_centroid(cluster)**: Calcula el centroide de un cluster.
- **add_text(text, name=None)**: Añade un nuevo texto al modelo, actualizando clusters y centroides.
- **predict(text)**: Predice a qué persona pertenece un nuevo texto.
- **save_model(filename)**: Guarda el modelo entrenado en un archivo.
- **load_model(filename)**: Carga un modelo entrenado desde un archivo.

### train.py

- **train_from_files(files, threshold)**: Entrena el modelo utilizando archivos de texto.

### utils.py

- **setup_logging()**: Configura el registro para el programa.

## Licencia

Este proyecto está bajo la Licencia MIT. Para más detalles, consulte el archivo `LICENSE`.