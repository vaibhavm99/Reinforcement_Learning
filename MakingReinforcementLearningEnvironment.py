'''
Blue = player
Green = food
Red = enemy
'''
import numpy as np 
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import cv2
import pickle
from matplotlib import style
import time
style.use("ggplot")
SIZE = 10 # it will be a 10 x 10 grid
HM_EPISODES = 25000 # How many episodes
MOVE_PENALTY = 1 # Penalty for moving(to be substracted)
ENEMY_PENALTY = 300 # Penalty for striking into enemy(to be substracted)
FOOD_REWARD = 25 # Reward to getting food
epsilon = 0.9 # Randomness factor(0 don't explore, 1 always explore)
EPS_DECAY = 0.9998 # Decay of epsilon
SHOW_EVERY = 3000
start_q_table = None # or a filename(if you have a q table and want to learn from that point)
LEARNING_RATE = 0.1
DISCOUNT = 0.95 # how important are future actions based on current actions
#Definitions for what those numbers represent in dictionary(keys)
PLAYER_N = 1
FOOD_N = 2
ENEMY_N = 3
# the dictionary(these numbers are colors in bgr format)
d = {1: (255, 175, 0),
     2: (0, 255, 0),
     3: (0, 0, 255)}
# Class for blob
class blob:
    def __init__(self):
        self.x = np.random.randint(0,SIZE)
        self.y = np.random.randint(0,SIZE)
    def __str__(self):
       return f"{self.x},{self.y}"
    def __sub__(self,other):
        return (self.x - other.x, self.y - other.y)  
    def action(self, choice):
        '''
        Gives us 4 total movement options. (0,1,2,3)
        Can only move diagonally, you can add other movements if you want
        '''
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)
    # Move randomly if no value passed or move specifically
    def move(self,x = False, y = False):
        # If no value for x, move randomly
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        # If no value for y, move randomly
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y
        # If we are out of bounds, fix!
        if self.x < 0:
            self.x = 0
        elif self.x > SIZE-1:
            self.x = SIZE-1
        if self.y < 0:
            self.y = 0
        elif self.y > SIZE-1:
            self.y = SIZE-1
# Create or load the q table
if start_q_table is None:
    q_table = {} # Initializing q table
    for x1 in range(-SIZE+1, SIZE):
        for y1 in range(-SIZE+1, SIZE):
            for x2 in range(-SIZE+1, SIZE):
                for y2 in range(-SIZE+1, SIZE):
                    q_table[((x1,y1), (x2,y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)
episode_rewards = []
for episode in range(HM_EPISODES):
    player = blob()
    food = blob()
    enemy = blob()
    if episode % SHOW_EVERY == 0:
        print(f"on #{episode}, epsilon is {epsilon}")
        print(f"{SHOW_EVERY} ep mean: {np.mean(episode_rewards[-SHOW_EVERY:])}")
        show = True
    else:
        show = False
    episode_reward = 0
    for i in range(200):
        obs = (player-food, player-enemy)
        #print(obs)
        if np.random.random() > epsilon:
            # GET THE ACTION
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        # Take the action!
        player.action(action)

        #### MAYBE ### (LET THE ENEMY AND FOOD MOVE)
        #enemy.move()
        #food.move()
        ##############

        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY
        ## NOW WE KNOW THE REWARD, LET'S CALC YO
        # first we need to obs immediately after the move.
        new_obs = (player-food, player-enemy)
        max_future_q = np.max(q_table[new_obs])
        current_q = q_table[obs][action]

        if reward == FOOD_REWARD:
            new_q = FOOD_REWARD
        else:
            new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[obs][action] = new_q
        if show: # same as env.render()
            env = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
            env[food.x][food.y] = d[FOOD_N]  # sets the food location tile to green color
            env[player.x][player.y] = d[PLAYER_N]  # sets the player tile to blue
            env[enemy.x][enemy.y] = d[ENEMY_N]  # sets the enemy location to red
            img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
            img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
            cv2.imshow("image", np.array(img))  # show it!
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:  # crummy code to hang at the end if we reach abrupt end for good reasons or not.
                if cv2.waitKey(500) & 0xFF == ord('q'):
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY: # GAME IS OVER
            break
    #print(episode_reward)
    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY
# PLOTTING THE STATS    
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY,))/SHOW_EVERY, mode='valid')
plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"Reward {SHOW_EVERY}ma")
plt.xlabel("episode #")
plt.show()
# Saving the q table
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)