import random
import torch 
import matplotlib.pyplot as plt
from torch import nn
from torchtext import data
from torchtext import datasets

SEED = 1234
MAX_VOCAB_SIZE = 25000
BATCH_SIZE = 64

torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_mask(src, idx):
	mask = (src != idx).unsqueeze(2)
	return mask

class Model(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, dropout):
		super(Model, self).__init__()
		self.embedding = nn.Embedding(input_size, hidden_size)

		self.fc1 = nn.Linear(hidden_size, hidden_size)
		self.bn1 = nn.BatchNorm1d(hidden_size)

		self.fc2 = nn.Linear(hidden_size, hidden_size)
		self.bn2 = nn.BatchNorm1d(hidden_size)

		self.fc3 = nn.Linear(hidden_size, hidden_size)
		self.bn3 = nn.BatchNorm1d(hidden_size)

		self.fc4 = nn.Linear(hidden_size, output_size)
		self.relu = nn.ReLU() 
		self.softmax = nn.Softmax(dim = 1)
		self.dropout = nn.Dropout(p=dropout)

	def forward(self, x, mask):
		embedded = self.dropout(self.embedding(x))
		embedded.masked_fill(mask == 0, 0)		
		x_ = embedded.mean(dim = 1)		
		x_ = self.relu(self.bn1(self.dropout(self.fc1(x_))))
		x_ = self.relu(self.bn2(self.dropout(self.fc2(x_))))
		x_ = self.relu(self.bn3(self.dropout(self.fc3(x_))))
		x_ = self.softmax(self.fc4(x_))
		return x_


TEXT = data.Field(tokenize="spacy", init_token = '<SOS>', eos_token = '<EOS>', lower = True, sequential = True, 
				pad_token = "<PAD>", batch_first = True, tokenizer_language='en')

LABEL = data.LabelField(dtype = torch.int64)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

TEXT.build_vocab(train_data, max_size = MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

train_iter, test_iter = data.BucketIterator.splits((train_data, test_data),
													batch_size=BATCH_SIZE, device=device, 
													shuffle=True, sort_key=lambda x: len(x.text),
													sort_within_batch=True)
input_size = len(TEXT.vocab)
hidden_size = 300
output_size = len(LABEL.vocab)
dropout = 0.3
PAD_IDX = TEXT.vocab.stoi[TEXT.pad_token]
epochs = 100

model = Model(input_size, hidden_size, output_size, dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
criterion = nn.CrossEntropyLoss().to(device)

def train():
	train_epoch_loss = 0
	correct = 0
	model.train()
	for _, batch in enumerate(train_iter):
		optimizer.zero_grad()
		src, tgt = batch.text, batch.label
		mask = create_mask(src, PAD_IDX)
		output = model(src, mask)
		loss = criterion(output, tgt)
		loss.backward()
		optimizer.step()

		train_epoch_loss += loss.item()
		predicted_labels = torch.argmax(output, 1)	
		correct += (predicted_labels == tgt).sum().item()

	acc = 100 * correct / len(train_data)
	return train_epoch_loss, acc

def test():
	epoch_loss = 0
	correct = 0
	model.eval()
	with torch.no_grad():
		for _, batch in enumerate(test_iter):
			src, tgt = batch.text, batch.label
			mask = create_mask(src, PAD_IDX)
			output = model(src, mask)
			loss = criterion(output, tgt)

			epoch_loss += loss.item()
			predicted_labels = torch.argmax(output, 1)	
			correct += (predicted_labels == tgt).sum().item()

	acc = 100 * correct / len(test_data)
	return epoch_loss, acc

train_loss_list, train_acc_list, test_loss_list, test_acc_list = [], [], [], []

print("Training Starts")
for i in range(epochs):
	train_loss, train_acc = train()
	test_loss, test_acc = test()

	print("Epoch:", i+1)
	print("Training Loss:", train_loss)
	print("Training Accuracy:", train_acc)
	print("Test Loss:", test_loss)
	print("Test Accuracy:", test_acc)
	print("*"*40)

	train_loss_list.append(train_loss)
	train_acc_list.append(train_acc)
	test_loss_list.append(test_loss)
	test_acc_list.append(test_acc)

fig = plt.figure()
ax1 = fig.add_subplot("121")
ax1.set_title("Accuracy")
ax1.plot(train_acc_list, label="train")
ax1.plot(test_acc_list, label="test")
ax1.legend()

ax2 = fig.add_subplot("122")
ax2.set_title("Loss")
ax2.plot(train_loss_list, label="train")
ax2.plot(test_loss_list, label="test")
ax2.legend()

plt.savefig("plot.png")
