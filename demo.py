from magent2.environments import battle_v4
import os
import cv2
import sys


if __name__ == "__main__":
    env = battle_v4.env(map_size=45, render_mode="rgb_array")
    env.reset()
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 35
    frames = []
    

    # pretrained policies
    frames = []
    env.reset()
    from base_model import QNetwork
    import torch

    q_network = QNetwork(
        env.observation_space("red_0").shape, env.action_space("red_0").n
    )
    q_network.load_state_dict(
        torch.load("model/red.pt", weights_only=True, map_location="cuda")
    )
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()
        

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "red":
                observation = (
                    torch.Tensor(observation).float().permute([2, 0, 1]).unsqueeze(0)
                )
                with torch.no_grad():
                    q_values = q_network(observation)
                action = torch.argmax(q_values, dim=1).numpy()[0]
            else:
                action = env.action_space(agent).sample()
        
        env.step(action)

        if agent == "red_0":
            frames.append(env.render())

'''
    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"pretrained-retest.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording pretrained agents")

    env.close()
'''