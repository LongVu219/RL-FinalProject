import matplotlib.pyplot as plt

import re


def plot_train_loss(
    path="./logs/log_error.txt",
    path_save="./logs/log_error.png"
):
    values = []
    pattern = r"Epoch #(?P<epoch_num>\d+), Batch #(?P<id>\d+): Loss=(?P<total_loss>\d+\.\d+)"
    with open(path, "r") as f:
        for line in f.readlines():
            match = re.match(pattern, line)
            if match:
                epoch_num = match.group("epoch_num")
                batch_id = match.group("id")
                total_loss = match.group("total_loss")
                values.append(total_loss)

    plt.plot(values)
    plt.title("MSE Loss plot")
    plt.savefig(path_save)
    plt.clf()
    plt.close()


def plot_eval(
    path="./logs/log_eval.txt",
    path_save="./logs/log_eval.png"
):
    values = []
    pattern = r"Average red vs blue power projection: (?P<avg>\d+\.\d+)"
    with open(path, "r") as f:
        for line in f.readlines():
            # Extract the variable from the line
            match = re.match(pattern, line)
            if match:
                avg = match.group("avg")
                print(f"Average: {avg}")
                values.append(avg)

    plt.plot(values)
    plt.title("Red vs Blue projection plot (the lower the better)")
    plt.savefig(path_save)
    plt.clf()
    plt.close()
# Regular expression pattern to match the format

def total_plot():
    plot_train_loss()
    plot_eval()

if __name__ == "__main__":
    total_plot()
