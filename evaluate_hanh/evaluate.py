import re

with open("out.txt", "r") as f:
    pattern_epoch_loss = r"Epoch number 50 loss : (\d+\.\d+)"
    pattern_pp = r"Average red vs blue power projection: (\d+\.\d+)"
    pattern_win_rate = r"Win rate : (\d+\.\d+) \| Lose rate : (\d+\.\d+) \| Draw rate : (\d+\.\d+)"

    match = re.search(pattern_epoch_loss, epoch_loss)

    if match:
        loss_value = match.group(1)
        print("Loss value:", loss_value)
