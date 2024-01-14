import seaborn as sns
import matplotlib.pyplot as plt

train_loss = [0.092188, 0.085606, 0.082553, 0.082288, 0.080601, 0.079573, 0.077640, 0.076990, 0.076229, 0.074265]
valid_loss = [0.112501, 0.085238, 0.068926, 0.066756, 0.077916, 0.083918, 0.066423, 0.063398, 0.061721, 0.060911]

sns.set_style("darkgrid")

sns.lineplot(x=range(len(train_loss)), y=train_loss, label='Train Loss')
sns.lineplot(x=range(len(valid_loss)), y=valid_loss, label='Valid Loss')

plt.xlabel('Epoch')
plt.ylabel('Loss')
#plt.title('Training and Validation Loss Over Epochs')
plt.legend()

plt.savefig('loss_plot.png')
plt.show()
