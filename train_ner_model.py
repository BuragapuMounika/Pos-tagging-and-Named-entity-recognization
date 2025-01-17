import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
from nltk.tokenize import word_tokenize

# Dummy data for training (replace with your actual data)
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
label_encoder.fit(["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"])  # Add your entity types accordingly

# Prepare training data
X_train = []
y_train = []

for sentence, label_seq in zip(sentences, labels):
    tokens = word_tokenize(sentence)
    indices = [word_to_index[token] for token in tokens]
    X_train.append(indices)
    y_train.append(label_encoder.transform(label_seq))

# Define the BiLSTM model
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

# Train the model
def train_model():
    model = BiLSTM(embedding_dim=100, hidden_dim=128, output_dim=len(label_encoder.classes_), num_layers=1)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert to tensor and pad sequences
    X_train_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in X_train], batch_first=True)
    y_train_tensor = torch.nn.utils.rnn.pad_sequence([torch.tensor(seq) for seq in y_train], batch_first=True)

    # Training loop
    model.train()
    for epoch in range(10):  # Number of epochs
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = loss_function(outputs.view(-1, len(label_encoder.classes_)), y_train_tensor.view(-1))
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

    # Save the model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved as model.pth")

# Uncomment to train the model and save it
# train_model()

