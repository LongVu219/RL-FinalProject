import re
import matplotlib.pyplot as plt
import numpy as np

ep_losses = []
pp_values = []
rates = []

with open("out.txt", "r") as f:
    pattern_epoch_loss = r"Epoch number 50 loss : (\d+\.\d+)"
    pattern_pp = r"Average red vs blue power projection: (\d+\.\d+)"
    pattern_win_rate = r"Win rate : (\d+\.\d+) \| Lose rate : (\d+\.\d+) \| Draw rate : (\d+\.\d+)"

    for line in f.readlines():
        match_ep_loss = re.search(pattern_epoch_loss, line.rstrip())
        match_pp = re.search(pattern_pp, line.rstrip())
        match_win_rate = re.search(pattern_win_rate, line.rstrip())

        if match_ep_loss:
            loss_value = match_ep_loss.group(1)
            ep_losses.append(float(loss_value))
        elif match_pp:
            power_projection_value = match_pp.group(1)
            pp_values.append(float(power_projection_value))
        elif match_win_rate:
            win_rate = match_win_rate.group(1)
            lose_rate = match_win_rate.group(2)
            draw_rate = match_win_rate.group(3)
            rates.append((float(win_rate), float(lose_rate), float(draw_rate)))

print(len(ep_losses))
print(len(pp_values))
print(len(rates))
pp_values = np.array(pp_values)
rates = np.array(rates)[:30, 0]

plt.plot(pp_values[:30])
plt.title("DQN: Power projection value (lower=better, save best)")
plt.xlabel("Episode #")
plt.ylabel("Power projection")

plt.savefig("test.png")

plt.clf()
plt.close()
