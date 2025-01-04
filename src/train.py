from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam

from src.model import SentimentRNN
from src.data_loader import create_dataloader

if __name__ == "__main__":
	train_loader, test_loader, vocab = create_dataloader()

	vocab_size = len(vocab) + 1
	embed_size = 128
	hidden_size = 128
	output_size = 2
	model = SentimentRNN(vocab_size, embed_size, hidden_size, output_size)

	criterion = nn.CrossEntropyLoss()  # Binary Classification
	optimizer = Adam(model.parameters(), lr=0.001)

	num_epochs = 10
	for epoch in range(num_epochs):
		model.train()
		epoch_loss = 0
		for texts, labels in train_loader:
			outputs = model(texts)
			loss = criterion(outputs, labels)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

			epoch_loss += loss.item()

		print(
			f"Epoch: [{epoch+1}/{num_epochs}], Loss: {epoch_loss / len(train_loader):.4f}"
		)

	try:
		model_path = Path("src/models/")		
		file_name = model_path / "sentiment_rnn.pth"
		torch.save(model.state_dict(), file_name)
		print("Model Saved successfully")
	except Exception as e:
		print(f"Unable to save model.\nEror: {e}")
