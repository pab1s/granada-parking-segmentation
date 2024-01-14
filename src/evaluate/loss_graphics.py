import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

csv_paths = [
    'results/logs/unparked_deeplab_granada.csv',
    'results/logs/parked_deeplab_granada.csv',
]
model_names = ['Unparked Cars Model', 'Parked Cars Model']

sns.set_style("darkgrid")
colors = ['b', 'g', 'r'] 

for csv_file, model_name, color in zip(csv_paths, model_names, colors):
    data = pd.read_csv(csv_file)

    train_loss = data['train_loss']
    valid_loss = data['valid_loss']

    sns.lineplot(x=range(len(train_loss)), y=train_loss, label=f'{model_name} Train Loss', color=color)
    sns.lineplot(x=range(len(valid_loss)), y=valid_loss, label=f'{model_name} Valid Loss', linestyle='--', color=color)

plt.xlabel('Epoch')
plt.ylabel('Loss')
# plt.title('Training and Validation Loss Over Epochs for Different Models')
plt.legend()

#plt.ylim(0, 0.5)

plt.savefig('parked_unparked_plot.png')
plt.show()
