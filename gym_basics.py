import gym
import random

# Criando ambiente
# env = gym.make('CartPole-v1')
env = gym.make('MountainCar-v0')

# Espaço Observado..
# São as informações do ambiente nesse estado
# esse exemplo do CartPole por exemplo, tem 4 informações:
# Observation: 
#     Type: Box(4)
#     Num	Observation                 Min         Max
#     0	    Cart Position             -4.8            4.8
#     1	    Cart Velocity             -Inf            Inf
#     2	    Pole Angle                 -24 deg        24 deg
#     3	    Pole Velocity At Tip      -Inf            Inf
print("Observation space:", env.observation_space)

# Espaço de ações
# São as ações que o agente pode realizar
# CartPole tem 2 ações:
# Actions:
#     Type: Discrete(2)
#     Num	Action
#     0	Push cart to the left
#     1	Push cart to the right
print("Action space:", env.action_space)


# Agente 
# Responsável de receber o estado do ambiente
# e realizar uma ação a partir disso
class Agent():
    def __init__(self, env):
        self.action_size = env.action_space.n

    def get_action(self, state):
        # Política: 
        # Ação baseada no ângulo
        # O agente recebe a informação do ambiente nesse estado
        # e se utiliza da informação do angulo do pendulo 
        # se o angulo estiver pendendo a um lado
        # o agente deve realizar a ação que 
        # movimenta ao lado oposto 
        #pole_angle = state[2]
        #action = 0 if pole_angle < 0 else 1

        # Ação aleatória
        action = random.choice(range(self.action_size))
        return action


# Loop principal

agent = Agent(env)
state = env.reset()

for _ in range(150):
    action = agent.get_action(state)
    state, reward, done, info = env.step(action)
    env.render()
env.close()
