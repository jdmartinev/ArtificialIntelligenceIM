# Modelo de Clasificación y Localización

El objetivo es construir y entrenar una red de clasificación y localización. Este ejercicio mostrará la flexibilidad del Deep Learning con varios tipos de salidas heterogéneas (cajas delimitadoras y clases).

Construiremos el modelo en tres pasos consecutivos:
- **Extraer anotaciones de etiquetas** de un conjunto de datos estándar de detección de objetos, en este caso, **Pascal VOC 2007**;
- Utilizar un modelo preentrenado de clasificación de imágenes (específicamente ResNet50) para **precomputar representaciones convolucionales** con forma `(7, 7, 2048)` para todas las imágenes en el conjunto de entrenamiento de detección de objetos;
- **Diseñar y entrenar un modelo base de detección de objetos con dos cabezas** para predecir:
  - etiquetas de clase (5 clases posibles)
  - coordenadas de la caja delimitadora de un solo objeto detectado en la imagen

Ten en cuenta que el modelo base simple presentado solo detectará una sola ocurrencia de una clase por imagen. Se necesitaría más trabajo para detectar todas las posibles ocurrencias de objetos en las imágenes. Consulta las diapositivas de la lección para referencias a modelos de detección de objetos de vanguardia como Faster RCNN y YOLO9000.

El notebook base lo pueden encontrar aquí: [AICompetition02](https://www.kaggle.com/code/juanmartinezv4399/aicompetition02)

Las representaciones pre-calculadas se obtuvieron con el siguiente script: [AICompetition02_comp_representations](https://www.kaggle.com/code/juanmartinezv4399/aicompetition02-comp-representations)
