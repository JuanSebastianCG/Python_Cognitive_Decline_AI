
# Cognitive Decline Prediction Model

## Tabla de Contenidos
- [Resumen del Proyecto](#resumen-del-proyecto)
- [Descripción Completa](#descripción-completa)
  - [Contexto](#contexto)
  - [Desafío Técnico](#desafío-técnico)
  - [Plan de Acción](#plan-de-acción)
- [Instalación](#instalación)
  - [Pre-requisitos](#pre-requisitos)
  - [Configuración del Entorno](#configuración-del-entorno)
- [Uso](#uso)
- [Contribución](#contribución)
  - [Cómo Contribuir](#cómo-contribuir)
  - [Código de Conducta](#código-de-conducta)
- [Licencia](#licencia)
- [Autores](#autores)
- [Agradecimientos](#agradecimientos)

## Resumen del Proyecto
Este proyecto desarrolla un modelo predictivo para la detección temprana del deterioro cognitivo en adultos mayores, implementado a través de microservicios para facilitar su uso y escalabilidad.

## Descripción Completa

### Contexto
Enfrentamos un aumento global en la población de adultos mayores, donde el deterioro cognitivo se está convirtiendo en una preocupación mayor. Este proyecto utiliza tecnologías de vanguardia para abordar este desafío significativo.

### Desafío Técnico
Diseñar un modelo que sea preciso y eficiente, capaz de procesar grandes volúmenes de datos clínicos para ofrecer diagnósticos tempranos y personalizados.

### Plan de Acción
1. **Preparación de Datos**: Limpieza, estandarización, y análisis preliminar.
2. **Desarrollo del Modelo**: Selección de características y modelado predictivo.
3. **Validación**: Técnicas de validación cruzada para asegurar la fiabilidad del modelo.
4. **Optimización**: Mejoras en el modelo mediante técnicas de optimización avanzadas.
5. **Monitoreo y Actualización**: Implementación de sistemas para el monitoreo continuo del rendimiento del modelo.
6. **Despliegue**: Microservicios utilizando FastAPI y Docker.

## Instalación

### Pre-requisitos
- Python 3.8+
- pip
- virtualenv
- docker

```bash
# Clona el repositorio
git clone https://github.com/juancgiraldo/cognitive-decline-prediction.git
cd cognitive-decline-prediction

# Configura el entorno virtual
python -m venv venv
source venv/bin/activate

# Instala las dependencias
pip install -r requirements.txt
```

## Configuración del Entorno

### Configuración de IDE
Se recomienda utilizar Visual Studio Code con las siguientes extensiones para un entorno de desarrollo eficiente:
- Python
- Docker
- GitLens

### Creación del Entorno Python
Utilizaremos `virtualenv` para crear un entorno aislado que permita manejar las dependencias de forma segura y eficaz.

```bash
    

```

## Uso

### Ejecución del Modelo
Para iniciar el servidor y exponer el modelo como un microservicio, ejecute:

1. **configurar variables de entorno**: 

```bash
# .env 
# Variables de entorno
DEBUG=True
HOST=
PORT= 

```

2. **Execute docker**: 
```bash
docker-compose up
```

```bash

### Uso del API
Una vez el servidor esté funcionando, puedes interactuar con el modelo a través de la API generada por FastAPI, accediendo a la documentación interactiva en `http://localhost:8000/docs`.



## Licencia

## Autores
- **Juan Sebastian Cardenas Giraldo** - *Trabajo inicial* - [juancgiraldo](https://github.com/juancgiraldo)

## Agradecimientos
- A todos los colaboradores y mentores que han participado en este proyecto.
- A la Universidad Autónoma de Manizales por el soporte y los recursos proporcionados.
