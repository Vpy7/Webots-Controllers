import time
import csv
import os
import numpy as np
from deepbots.supervisor.controllers.robot_supervisor_env import RobotSupervisorEnv
from utilities import normalize_to_range
from PPO_agent import PPOAgent, Transition
from gym.spaces import Box, Discrete
from controller import Robot

# Definición de los límites del entorno para la posición (x, y) y otros parámetros
xlimits = [-4.5, 4.5]
ylimits = [-3, 3]
glimits = [-0.8, 0.8]

# Configuración inicial para el LiDAR: 360 medidas con valores mínimos y máximos
lidar_min = np.zeros(360)
lidar_max = np.full(360, 1e3)

class TBot3(RobotSupervisorEnv):
    def __init__(self):
        super().__init__()
        
        # Definición del espacio de observación: [x, y, vx, vy, ángulo, 360 valores LiDAR]
        self.observation_space = Box(
            low=np.array([xlimits[0], ylimits[0], -1e3, -1e3, 0, *lidar_min]),
            high=np.array([xlimits[1], ylimits[1], 1e3, 1e3, 2*np.pi, *lidar_max]),
            dtype=np.float64
        )
        # Definición del espacio de acción: 3 acciones discretas
        self.action_space = Discrete(3)

        # Inicialización del robot y la unidad inercial para obtener orientación
        self.robot = self.getSelf()
        self.orientation = self.getDevice("inertial unit")
        self.orientation.enable(self.timestep)

        # Configuración del LiDAR y activación del point cloud
        self.lidar = self.getDevice("LDS-01")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        self.lidar_width = self.lidar.getHorizontalResolution()
        self.lidar_max_range = self.lidar.getMaxRange()

        # Referencia a la pelota del entorno
        self.ball = self.getFromDef("BALL")

        # Inicialización de los motores de las ruedas y configuración de su velocidad inicial
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setVelocity(0.0)
        self.left_motor.setVelocity(0.0)

        # Parámetros para controlar la duración de cada episodio y almacenar puntajes
        self.steps_per_episode = 5000
        self.episode_score = 0
        self.episode_score_list = []

    def get_default_observation(self):
        """
        Retorna un vector de observación por defecto compuesto de ceros.
        El tamaño es la suma de 5 variables base y los 360 valores del LiDAR (365 en total).
        """
        return np.zeros(365)

    def get_observations(self):
        """
        Obtiene las observaciones actuales del robot y las normaliza:
          - Posición (x, y) normalizada entre -1 y 1.
          - Velocidad (x, y) normalizada entre -1 y 1.
          - Ángulo de orientación normalizado entre -1 y 1.
          - Valores del LiDAR normalizados entre -1 y 1.
        Retorna un vector con todas estas observaciones.
        """
        robot_xpos = normalize_to_range(self.robot.getPosition()[0], xlimits[0], xlimits[1], -1, 1)
        robot_ypos = normalize_to_range(self.robot.getPosition()[1], xlimits[0], xlimits[1], -1, 1)
        robot_xvel = normalize_to_range(self.robot.getVelocity()[0], -6.67, 6.67, -1, 1)
        robot_yvel = normalize_to_range(self.robot.getVelocity()[1], -6.67, 6.67, -1, 1)
        robot_angle = normalize_to_range(self.orientation.getRollPitchYaw()[2], -2*np.pi, 2*np.pi, -1, 1)
        
        # Se vectoriza la función de normalización para aplicarla al LiDAR
        normalize_vectorized = np.vectorize(normalize_to_range)
    
        # Obtención y normalización de los valores del LiDAR
        lidar_values = self.lidar.getRangeImage()
        if lidar_values is None:
            lidar_values = np.zeros(self.lidar.getHorizontalResolution())
        else:
            # Sustituye valores NaN o infinitos por números finitos
            lidar_values = np.nan_to_num(lidar_values, nan=10, posinf=10, neginf=-10)
        lidar_values = normalize_vectorized(lidar_values, -10, 10, -1, 1)
        
        # Combina todas las variables en un único vector de observación
        observations = np.array(
            [robot_xpos, robot_ypos, robot_xvel, robot_yvel, robot_angle, *lidar_values],
            dtype=np.float64
        )
        return observations
                         
    def get_reward(self, action=None):
        """
        Calcula la recompensa actual basada en la posición y velocidad de la pelota y del robot.
        Se incentiva que la pelota se acerque a la meta (x=4.5) y se penaliza si:
          - El robot o la pelota se acercan a los límites del campo.
          - La pelota se detiene.
        """
        target_x, target_y_min, target_y_max = 4.5, -0.8, 0.8

        ball_x, ball_y = self.ball.getPosition()[0], self.ball.getPosition()[1]
        ball_velx, ball_vely = self.ball.getVelocity()[0], self.ball.getVelocity()[1]
        robot_x, robot_y = self.robot.getPosition()[0], self.robot.getPosition()[1]
        
        distance = np.sqrt((ball_x - target_x)**2 + (ball_y)**2)
        reward = 1/(distance + 1e-10)**2
        
        margin = 0
        if (robot_x <= xlimits[0] + margin or robot_x >= xlimits[1] - margin or
            robot_y <= ylimits[0] + margin or robot_y >= ylimits[1] - margin):
            reward -= 1000
        if (ball_x <= xlimits[0] + margin or ball_x >= xlimits[1] - margin or
            ball_y <= ylimits[0] + margin or ball_y >= ylimits[1] - margin):
            reward -= 10000
        if ball_x >= 4.5 and target_y_min <= ball_y <= target_y_max:
            reward += 100000
        if ball_velx == 0 or ball_vely == 0:
            reward -= 10
            
        return reward

    def is_done(self):
        """
        Determina si el episodio ha terminado (la pelota sale de los límites).
        """
        ball_x, ball_y = self.ball.getPosition()[0], self.ball.getPosition()[1]
        if not (xlimits[0] <= ball_x <= xlimits[1] and ylimits[0] <= ball_y <= ylimits[1]):
            return True
        return False
        
    def solved(self):
        """
        Evalúa si la tarea ha sido resuelta según el desempeño en episodios anteriores.
        """
        if len(self.episode_score_list) > 50000:
            if np.mean(self.episode_score_list[-100:]) > 500.0:
                return True
        return False
        
    def get_info(self):
        """
        Retorna información adicional del entorno.
        """
        return {}
    
    def render(self, mode='human'):
        pass
        
    def apply_action(self, action):
        """
        Ejecuta la acción indicada modificando la velocidad de las ruedas:
          0: Avanzar recto.
          1: Giro suave a la izquierda.
          2: Giro suave a la derecha.
        """
        base_speed = 6.0
        if action[0] == 0:
            left_speed = base_speed
            right_speed = base_speed
        elif action[0] == 1:
            left_speed = -base_speed
            right_speed = base_speed
        elif action[0] == 2:
            left_speed = base_speed
            right_speed = -base_speed 
        
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)
            
# ======================= BLOQUE PRINCIPAL =======================
env = TBot3()
agent = PPOAgent(number_of_inputs=env.observation_space.shape[0],
                 number_of_actor_outputs=env.action_space.n)

solved = False
episode_count = 0
episode_limit = 5000

checkpoint_path = "forward_ppo"

if os.path.exists(checkpoint_path + "_actor.pkl") and os.path.exists(checkpoint_path + "_critic.pkl"):
    agent.load(checkpoint_path)
    print("Checkpoint loaded from", checkpoint_path)
else:
    print("No checkpoint found. Training from scratch.")

while not solved and episode_count < episode_limit:
    # Reinicia el entorno para comenzar un nuevo episodio
    observation = env.reset()
    env.episode_score = 0  

    # Bucle interno que recorre cada paso dentro del episodio
    for step in range(env.steps_per_episode):
        # El agente selecciona una acción basándose en la observación actual
        selected_action, action_prob = agent.work(observation, type_="selectAction")
        new_observation, reward, done, info = env.step([selected_action])

        # Crea y almacena la transición para entrenamiento posterior
        trans = Transition(observation, selected_action, action_prob, reward, new_observation)
        agent.store_transition(trans)

        # Actualiza la puntuación acumulada del episodio
        env.episode_score += reward  
        observation = new_observation  

        # Si se cumple la condición de término del episodio, finaliza el ciclo interno
        if done:
            env.episode_score_list.append(env.episode_score)
            agent.train_step(batch_size=step + 1)  # Ejecuta un paso de entrenamiento
            solved = env.solved()  # Verifica si el desempeño cumple con la condición de solución
            break

    print(f"Episode #{episode_count} finished with score: {env.episode_score} FORWARD")
    episode_count += 1

    # Guarda el estado del agente cada 100 episodios
    if episode_count % 100 == 0:
        agent.save(checkpoint_path)
        print("Checkpoint saved at episode", episode_count)

if not solved:
    print("Task is not solved, deploying agent for testing...")
elif solved:
    print("Task is solved, deploying agent for testing...")

observation = env.reset()
env.episode_score = 0.0

while True:
    selected_action, action_prob = agent.work(observation, type_="selectActionMax")
    observation, _, done, _ = env.step([selected_action])
    if done:
        observation = env.reset()
