import torch
import torch.nn as nn
import torch.optim as optim
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder

# Dummy data for training
sentences = [
    "Barack Obama was born in Hawaii.",
    "Microsoft Corporation is located in Redmond.",
    "New York City is a big city."
]
labels = [
    ["B-PER", "I-PER", "O", "O", "O"],
    ["B-ORG", "I-ORG", "O", "O"],
    ["B-LOC", "I-LOC", "O", "O"]
]

# Tokenization and label encoding
word_to_index = {word: idx + 1 for idx, word in enumerate(set(word for sentence in sentences for word in word_tokenize(sentence)))}
label_encoder = LabelEncoder()
label_encoder.fit(["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"])

# Prepare training data
X_train = []
y_train = []
for sentence, label_seq in zip(sentences, labels):
    tokens = word_tokenize(sentence)
    indices = [word_to_index[token] for token in tokens]
    X_train.append(indices)
    y_train.append(label_encoder.transform(label_seq))

# Define BiLSTM Model
class BiLSTM(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim, num_layers):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(len(word_to_index) + 1, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# Initialize the model
embedding_dim = 100
hidden_dim = 128
output_dim = len(label_encoder.classes_)
num_layers = 1
model = BiLSTM(embedding_dim, hidden_dim, output_dim, num_layers)

# Define the optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training the model (for demonstration purposes, we will keep it simple)
X_train_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in X_train], batch_first=True)
y_train_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in y_train], batch_first=True)

model.train()
for epoch in range(5):  # Small number of epochs for demonstration
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs.view(-1, output_dim), y_train_tensor.view(-1))
    loss.backward()
    optimizer.step()
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
print("Model saved as model.pth")
