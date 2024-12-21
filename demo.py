from magent2.environments import battle_v4
import os
import cv2
import sys
from utils import *


demo_env = battle_v4.env(map_size=45, max_cycles = 300, render_mode="rgb_array")
demo_env.reset()
vid_dir = "video"
os.makedirs(vid_dir, exist_ok=True)
fps = 24
frames = []


# pretrained policies
from base_model import *
from train_episode import *

device = 'cuda'
blue_agent = get_agent('model/agent_base_final_kamikaze.pth', Resnet)
red_agent = get_agent('model/red.pt', QNetwork_final)

red_cnt = 81
blue_cnt = 81
mem = {}
step = 0

for agent in demo_env.agent_iter():

    observation, reward, termination, truncation, info = demo_env.last()
    
    agent_handle = agent.split("_")[0]

    if (reward >= 4.2):
        if (agent_handle == 'blue'): red_cnt -= 1
        else: blue_cnt -= 1

    if termination or truncation:
        action = None  # this agent has died
    else:
        if agent_handle == "blue":
            action = get_action(demo_env, None, agent, observation, blue_agent, 'best')
        else:
            action = get_action(demo_env, None, agent, observation, red_agent, 'best')
    
    demo_env.step(action)
    if (step%162 == 0):
        frames.append(demo_env.render())
    step += 1

print(red_cnt, blue_cnt)

height, width, _ = frames[0].shape
out = cv2.VideoWriter(
    os.path.join(vid_dir, f"battle_vs_best.mp4"),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (width, height),
)
for frame in frames:
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)
out.release()
print("Done recording battle !!!")

demo_env.close()
