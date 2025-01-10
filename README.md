# Clasificador de Traducciones ML vs Humanas

Este proyecto implementa un clasificador basado en RoBERTa para distinguir entre traducciones realizadas por máquina (ML) y traducciones realizadas por humanos profesionales.

## Descripción

El proyecto utiliza el modelo pre-entrenado RoBERTa-base-bne (Spanish) con fine-tuning específico para la tarea de clasificación de traducciones. El modelo es capaz de analizar textos en español y clasificarlos en dos categorías:
- Traducción ML (0): Textos traducidos por sistemas de traducción automática
- Traducción Humana (1): Textos traducidos por traductores profesionales

## Estructura del Proyecto

```
project-3-nlp/
│
├── data/
│   ├── TRAINING_DATA.txt     # Datos de entrenamiento para fine-tuning
│   ├── REAL_DATA.txt         # Datos para clasificar
│   └── resultados_clasificacion.csv  # Resultados de la clasificación
│
├── models/
│   └── roberta_finetuned/    # Modelo fine-tuned guardado
│
├── src/
│   └── notebooks/
│       ├── 01_exploratory_analysis.ipynb  # Análisis exploratorio
│       └── Reporte.ipynb     # Reporte de resultados
│
└── requirements.txt          # Dependencias del proyecto
```

## Requisitos

Para ejecutar este proyecto, necesitas:

```
torch
transformers
pandas
numpy
```

## Instalación

1. Clona el repositorio:
```bash
git clone [URL-del-repositorio]
cd project-3-nlp
```

2. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

1. **Preparación de datos**:
   - Coloca los textos a clasificar en `data/REAL_DATA.txt`
   - Un texto por línea

2. **Clasificación**:
   - Abre el notebook `src/notebooks/01_exploratory_analysis.ipynb`
   - Ejecuta las celdas para cargar el modelo y realizar predicciones
   - Los resultados se guardarán en `data/resultados_clasificacion.csv`

## Modelo

El proyecto utiliza RoBERTa-base-bne, un modelo de lenguaje pre-entrenado para español, con fine-tuning específico para la tarea de clasificación de traducciones. El modelo fine-tuned se guarda localmente en la carpeta `models/roberta_finetuned/`.

## Resultados

Los resultados de la clasificación incluyen:
- Texto original
- Texto procesado
- Predicción (0 o 1)
- Tipo de traducción (ML o Humana)

## Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue primero para discutir los cambios que te gustaría hacer.

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo LICENSE para más detalles. 