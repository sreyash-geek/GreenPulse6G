# Importing necessary libraries
import numpy as np
import gym
from gym import spaces
import random
import tensorflow as tf
import pygame
import gc
from collections import deque
import math
import torch
import torch.nn as nn
import torch.optim as optim

# Function for Buildings (Simulation of obstacles)
def line_intersects_rect(p1, p2, rect):
    # Coordinates : (x_min, y_min, x_max, y_max)
    def line_intersect(a, b, c, d):
        def ccw(u, v, w):
            return (w[1]-u[1]) * (v[0]-u[0]) > (v[1]-u[1]) * (w[0]-u[0])
        return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)
    
    (x_min, y_min, x_max, y_max) = rect
    edges = [
        ((x_min, y_min), (x_max, y_min)),
        ((x_max, y_min), (x_max, y_max)),
        ((x_max, y_max), (x_min, y_max)),
        ((x_min, y_max), (x_min, y_min))
    ]
    for edge in edges:
        if line_intersect(p1, p2, edge[0], edge[1]):
            return True
    return False

# Simulation Environment
class BaseStationEnv(gym.Env):
    """
    Advanced simulation environment for a 6G base station with realistic propagation.
    
    State (7-dimensional):
      1. Network load (0 to 100)
      2. BS mode (0: off, 1: sleep, 2: active)
      3. Average channel quality (0 to 100)
      4. Good channel ratio (fraction of UEs with quality > channel threshold)
      5. Average UE distance from BS (0 to 240)
      6. Average throughput (Mbps, 0 to 50)
      7. Average interference level 
    
    Propagation Model:
      - Log-distance path loss: PL = 30 + 10*n*log10(d) + shadowing, with n=3.5, shadowing ~ N(0,8).
      - LOS blockage: if the line from BS to UE intersects an obstacle (building), adding 20 dB loss.
      - Multipath fading: Rayleigh fading loss = 20*log10(r) where r ~ Rayleigh(scale=1).
      - Rain attenuation: if raining, add 5 dB loss.
      - Received power: Pr = Pt - (PL + extra losses). Pt = 46 dBm.
      - Channel quality: mapped linearly from Pr: 0 quality at ≤ -90 dBm, 100 quality at ≥ -60 dBm.
    
    Throughput:
      - Computed Throughput = 50 * log2(1 + avg_channel/100). 
    
    Obstacles:
      - Two fixed buildings in the environment.
    
    Actions:
      - 0: No change, 1: Sleep, 2: Active, 3: Off.
    
    Reward:
      - Composite reward that penalizes energy consumption and rewards service quality.
    """
    
    # Initialize all simulation parameters
    def __init__(self, max_steps=1000, num_UEs=20, channel_threshold=60):
        super(BaseStationEnv, self).__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.num_UEs = num_UEs
        self.channel_threshold = channel_threshold
        
        # State: [load, mode, avg_channel, good_ratio, avg_distance, avg_throughput, avg_interference]
        low = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([100, 2, 100, 1, 240, 50, 10], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
        self.action_space = spaces.Discrete(4)
        self.energy_cost = {0: 0, 1: 3, 2: 10}
        self.load_threshold = 60
        
        self.cumulative_energy = 0
        self.cumulative_reward = 0
        
        # Pygame parameters.
        self.viewer_initialized = False
        self.screen_width = 800
        self.screen_height = 800
        self.env_scale = self.screen_width / 240.0  # Maps [-120,120] to screen
        
        self.obstacles = [
            (-60, 10, -20, 40),
            (20, -80, 70, -40)
        ]
        
        # Determine rain condition at reset (Probability of rain with 5 dB attenuation).
        self.rain_attenuation = 0
        
        self.reset()
    
    # Calculate recieved power
    def _compute_received_power(self, pos):
        d = np.linalg.norm(pos)
        d = max(d, 1.0)
        # Shadowing
        shadowing = np.random.normal(0, 8)
        n = 3.5
        PL = 30 + 10 * n * np.log10(d) + shadowing
        extra_loss = 0
        for obs in self.obstacles:
            if line_intersects_rect((0,0), pos, obs):
                extra_loss += 20
                break
        # Multipath fading (Rayleigh)
        r = np.random.rayleigh(scale=1.0)
        fading_loss = 20 * np.log10(r)
        total_loss = PL + extra_loss + self.rain_attenuation + fading_loss
        Pt = 46  # dBm
        Pr = Pt - total_loss
        return Pr
    
    # Function to map power to channel quality
    def _map_power_to_quality(self, Pr):
        if Pr <= -90:
            return 0.0
        elif Pr >= -60:
            return 100.0
        else:
            return 100 * (Pr + 90) / 30
    
    # Calculate all channel related metrics
    def _compute_channel_metrics(self):
        qualities = []
        distances = []
        interferences = []
        for pos in self.ue_positions:
            d = np.linalg.norm(pos)
            distances.append(d)
            Pr = self._compute_received_power(pos)
            quality = self._map_power_to_quality(Pr)
            qualities.append(quality)
            # Rain Attenuation + Random Interference
            bg_interference = random.uniform(0, 10)
            total_interference = self.rain_attenuation + bg_interference
            interferences.append(total_interference)
        qualities = np.array(qualities)
        avg_channel = np.mean(qualities)
        good_ratio = np.mean(qualities > self.channel_threshold)
        avg_distance = np.mean(distances)
        avg_interference = np.mean(interferences)
        return avg_channel, good_ratio, avg_distance, avg_interference
    
    # Calculate Throughput
    def _compute_throughput(self, avg_channel):
        throughput = 50 * math.log2(1 + avg_channel/100)
        return throughput
    
    # Step parameters of an episode - Action to be performed by agent
    def step(self, action):
        done = False
        info = {}
        self.current_step += 1
        
        # Update BS mode.
        if action == 1:
            self.mode = 1
        elif action == 2:
            self.mode = 2
        elif action == 3:
            self.mode = 0
        
        # Simulate network load.
        period = 200
        base_load = 50 + 30 * np.sin(2 * math.pi * self.current_step / period)
        noise = np.random.normal(0, 5)
        self.load = np.clip(base_load + noise, 0, 100)
        
        energy = self.energy_cost[self.mode]
        self.cumulative_energy += energy
        
        avg_channel, good_ratio, avg_distance, avg_interference = self._compute_channel_metrics()
        avg_throughput = self._compute_throughput(avg_channel)
        
        # Reward function.
        base_reward = -energy
        if self.mode == 2:
            service_reward = 0.1 * self.load * (avg_channel / 100)
        elif self.mode == 1:
            service_reward = 0.05 * self.load * (avg_channel / 100)
        else:
            service_reward = 0
        
        # Reward components:
        baseline_cost = 20  # Simulated Fixed operational cost per step.
        throughput_bonus = avg_throughput  # Reward increases with higher throughput.
        service_bonus = (avg_channel * good_ratio * self.load / 100)  # Overall service quality.
        green_efficiency = throughput_bonus / (energy + 1)  # Efficiency of energy use.
        
        # Combining and weighting rewards (priority-based)
        reward = base_reward + service_reward + 0.3 * throughput_bonus + 0.4 * green_efficiency + 0.3 * service_bonus - baseline_cost
        
        # Additional penalization:
        if self.load > self.load_threshold and self.mode != 2:
            reward -= 30  # Heavy penalty if load is high but BS is not active.
        if self.load < 40 and self.mode == 2:
            reward -= 10  # Penalty if BS is active when load is low.
        if avg_channel < 80:
            reward -= (80 - avg_channel) * 0.5  # Penalize low channel quality.
        if throughput_bonus < 20:
            reward -= (20 - throughput_bonus) * 0.5  # Penalize low throughput.
        if self.mode == 0 and self.load > self.load_threshold:
            reward -= (self.load - self.load_threshold) * 2  # Extra penalty if BS is off during high load.

        self.cumulative_reward += reward
        
        # UE positions using a random walk.
        delta = np.random.normal(0, 1, (self.num_UEs, 2))
        self.ue_positions += delta
        self.ue_positions = np.clip(self.ue_positions, -120, 120)
        
        state = np.array([self.load, self.mode, avg_channel, good_ratio, avg_distance, avg_throughput, avg_interference], dtype=np.float32)
        if self.current_step >= self.max_steps:
            done = True
        
        return state, reward, done, info
    
    # Reset the environment at the end of every episode
    def reset(self):
        self.current_step = 0
        self.mode = 2
        self.load = 50
        self.cumulative_energy = 0
        self.cumulative_reward = 0
        self.ue_positions = np.random.uniform(-100, 100, (self.num_UEs, 2))
        self.rain_attenuation = 5 if random.random() < 0.3 else 0
        avg_channel, good_ratio, avg_distance, avg_interference = self._compute_channel_metrics()
        avg_throughput = self._compute_throughput(avg_channel)
        state = np.array([self.load, self.mode, avg_channel, good_ratio, avg_distance, avg_throughput, avg_interference], dtype=np.float32)
        return state
    
    # Visualization function using pygame rendering during live training
    # Whatever happens in a step, needs to be updated accordingly in the panel
    def render(self, mode='human'):
        if not self.viewer_initialized:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Advanced 6G Environment Simulation Environment - Prototype")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 24)
            self.viewer_initialized = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("[LOG] Quit event received; ignoring during training.")
        
        self.screen.fill((10, 10, 40))
        grid_color = (50, 50, 80)
        for x in range(0, self.screen_width, 40):
            pygame.draw.line(self.screen, grid_color, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, 40):
            pygame.draw.line(self.screen, grid_color, (0, y), (self.screen_width, y))
        
        center = (self.screen_width // 2, self.screen_height // 2)
        if self.mode == 2:
            bs_color = (0, 255, 0)
        elif self.mode == 1:
            bs_color = (255, 255, 0)
        else:
            bs_color = (255, 0, 0)
        pygame.draw.circle(self.screen, bs_color, center, 15)
        pygame.draw.circle(self.screen, (255, 255, 255), center, 5)
        
        # Obstacles.
        for obs in self.obstacles:
            x_min, y_min, x_max, y_max = obs
            rect = pygame.Rect(int(center[0] + x_min * self.env_scale),
                               int(center[1] - y_max * self.env_scale),
                               int((x_max - x_min) * self.env_scale),
                               int((y_max - y_min) * self.env_scale))
            pygame.draw.rect(self.screen, (100, 100, 100), rect)
        
        avg_channel, good_ratio, avg_distance, avg_interference = self._compute_channel_metrics()
        avg_throughput = self._compute_throughput(avg_channel)
        
        for pos in self.ue_positions:
            screen_x = int(center[0] + pos[0] * self.env_scale)
            screen_y = int(center[1] - pos[1] * self.env_scale)
            Pr = self._compute_received_power(pos)
            quality = self._map_power_to_quality(Pr)
            quality_norm = np.clip(quality / 100, 0, 1)
            ue_color = (int(255 * (1 - quality_norm)), int(255 * quality_norm), 0)
            pygame.draw.circle(self.screen, ue_color, (screen_x, screen_y), 8)
        
        # Environment metrics (Updation at top-right panel)
        info_lines = [
            f"Step: {self.current_step}/{self.max_steps}",
            f"Network Load: {self.load:.1f}",
            f"BS Mode: {self.mode} (2:active, 1:sleep, 0:off)",
            f"Cummulative Reward: {self.cumulative_reward:.2f}",
            f"Cummulative Energy: {self.cumulative_energy}",
            f"Avg. Channel Quality: {avg_channel:.1f}",
            f"Good Channels Ratio: {good_ratio*100:.1f}%",
            f"Avg. Distance: {avg_distance:.1f}",
            f"Throughput: {avg_throughput:.2f}",
            f"Avg. Interference: {avg_interference:.2f}"
        ]

        y_offset = 10
        for line in info_lines:
            text_surface = self.font.render(line, True, (255, 255, 255))
            self.screen.blit(text_surface, (10, y_offset))
            y_offset += 20
        
        pygame.display.flip()
        self.clock.tick(30)
    
    # Stop simulation rendering
    def close(self):
        if self.viewer_initialized:
            pygame.quit()
            self.viewer_initialized = False


# Define the DRL Agent (DQN) using PyTorch
class DQN(nn.Module):
    # Initialize the network architecture
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 24)
        self.fc5 = nn.Linear(24, 24)
        self.out = nn.Linear(24, action_size)
    
    # Forward pass
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.out(x)

# A DQN Control Agent (Base Station)
class DQNAgent:
    # Initialize the simulation parameters
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 64
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
    
    # Saving the tuple to memory of the agent
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # Take an action
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.action_size)
            return action
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state_tensor).cpu().numpy()[0]
        action = np.argmax(q_values)
        return action
    
    # Use the batched experience replay into memory, used by the control agent to make better decisions
    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        
        states = torch.FloatTensor([m[0] for m in minibatch]).to(self.device)
        actions = torch.LongTensor([m[1] for m in minibatch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([m[2] for m in minibatch]).to(self.device)
        next_states = torch.FloatTensor([m[3] for m in minibatch]).to(self.device)
        dones = torch.FloatTensor([float(m[4]) for m in minibatch]).to(self.device)
        
        current_q = self.model(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_q = self.model(next_states).max(1)[0]
        target_q = rewards + self.gamma * next_q * (1 - dones)
        
        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Training Loop with Live Environment Rendering
env = BaseStationEnv(max_steps=500, num_UEs=10)
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = DQNAgent(state_size, action_size)
episodes = 1000

# Collect episode rewards
episode_rewards = []

# Run episodes, where the agent will learn to behave appropriately from its action space
for e in range(episodes):
    state = env.reset()
    total_reward = 0
    print(f"\n[EPISODE {e+1} START] Environment reset.")
    for t in range(env.max_steps):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        env.render(mode='human')
        if done:
            break
        agent.replay()
    episode_rewards.append(total_reward)
    print(f"[EPISODE {e+1} SUMMARY] Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}")
    gc.collect()

env.close()
