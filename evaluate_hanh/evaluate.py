import re
import matplotlib.pyplot as plt
import numpy as np

pp_values = []
episode_id = []

with open("log.txt", "r") as f:
    pattern_pp = r"Avg projection power at episode (\d+): (\d+\.\d+)"
    for line in f.readlines():
        match = re.search(pattern_pp, line.rstrip())

        if match:
            episode = match.group(1)
            projection_power = match.group(2)
            print("Episode:", episode)
            print("Projection power:", projection_power)

            episode_id.append(int(episode))
            pp_values.append(float(projection_power))

print(len(pp_values))
pp_values = np.array(pp_values)
episode_id = np.array(episode_id, dtype=np.int64)

plt.plot(episode_id, pp_values)
plt.title("DQN + CrossQ: Power projection value (lower=better, save best)")
plt.xlabel("Episode #")
plt.ylabel("Power projection")

plt.savefig("test.png")
