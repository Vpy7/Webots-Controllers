# Simulación y Control de TurtleBot3 con Webots y ROS2

## Descripción

Este proyecto implementa un entorno de simulación para el entrenamiento y control de un **TurtleBot3 Burger** mediante **aprendizaje por refuerzo** con **PPO**. Utiliza **Webots** para la simulación, **Deepbots** para la integración con modelos de IA y **ROS2** para la transferencia de comportamientos a un robot físico.

## Tecnologías

- **Webots**: Simulación de entornos 3D para robots.
- **Deepbots**: Interfaz entre Webots y aprendizaje por refuerzo.
- **ROS2**: Middleware para el control del TurtleBot3 en hardware real.
- **PPO (Proximal Policy Optimization)**: Algoritmo de aprendizaje por refuerzo.

## Requisitos

- [Webots R2023b o superior](https://cyberbotics.com/#download)
- [Python 3.x.x](https://www.python.org/downloads/)
- [Deepbots v1.0.0](https://github.com/aidudezzz/deepbots)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [ROS2 Humble Hawksbill](https://docs.ros.org/en/humble/Installation.html)
- [Paquetes de ROS2 para TurtleBot3](https://github.com/ROBOTIS-GIT/turtlebot3)

## Estructura del Proyecto



- `controlador.py`: Controlador del robot en Webots.
- `PPO_agent.py`: Implementación del agente PPO.
- `utilities.py`: Funciones auxiliares para el entrenamiento.
- `ros_controller.py`: Nodo ROS2 para desplegar el modelo en el TurtleBot3 real. (Por Actualizar)

## Personalización

Se proporciona el Directorio de proyecto `TBOT3_FOOTBALL`, que contiene un ejemplo de uso de Webots, simulando un Partido de Futbol entre 4 TurtleBot3. Éste contiene el archivo `Mundo.wbt` que configura el entorno, así como el controlador `controlador.py` para el delantero. El manejo del resto de controladores queda aa discreción del usuario

Puedes modificar:

- **Entorno** (`Mundo.wbt`).
- **Aparato Lidar** ( Se proporciona actualización `RobotisLds02.proto`).
- **Sistema de recompensas** (`get_reward` en `controlador.py`).
- **Espacio de observación y acción** (`TBot3` en `controlador.py`).
- **Estrategia de entrenamiento** (`PPO_agent.py`).

## Referencias

- [Deepbots](https://github.com/aidudezzz/deepbots)
- [Webots](https://cyberbotics.com/)
- [Deepbots Tutorial en GitHub](https://github.com/aidudezzz/deepbots-tutorials/tree/master)
- [PPO: Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
- [ROS2](https://www.ros.org/)

---

Desarrollado por **Victor Leiva Espinoza** - Febrero 2025.

