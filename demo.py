from magent2.environments import battle_v4
import os
import cv2
import sys


if __name__ == "__main__":
    env = battle_v4.env(map_size=45, max_cycles = 2000, render_mode="rgb_array")
    env.reset()
    vid_dir = "video"
    os.makedirs(vid_dir, exist_ok=True)
    fps = 24
    frames = []
    

    # pretrained policies
    import base_model
    import resnet
    import torch

    q_network = resnet.QNetwork(
        env.observation_space("blue_0").shape, env.action_space("blue_0").n
    )
    q_network.load_state_dict(
        torch.load("model/agent_1.pth", weights_only=True, map_location="cuda")
    )
    for agent in env.agent_iter():

        observation, reward, termination, truncation, info = env.last()
        

        if termination or truncation:
            action = None  # this agent has died
        else:
            agent_handle = agent.split("_")[0]
            if agent_handle == "blue":
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


    height, width, _ = frames[0].shape
    out = cv2.VideoWriter(
        os.path.join(vid_dir, f"battle_best.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    out.release()
    print("Done recording battle !!!")

    env.close()
