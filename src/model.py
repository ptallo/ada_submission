import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from skimage import io

from clean.image import clean as clean_img
from clean.vocab import load_vocab
from dataset import CocoAnnotationDataset


class ImageCaptioningNet(nn.Module):
	def __init__(self, vocab_size, hidden_size=128):
		super(ImageCaptioningNet, self).__init__()
		self.hidden_size = hidden_size
		self.vocab_size = vocab_size
		self.final_channels = 50
		self.final_linear = 100

		self.conv1 = nn.Conv2d(3, 20, 11, 1)
		self.avg_pool1 = nn.AvgPool2d(3)
		self.conv2 = nn.Conv2d(20, 40, 11, 1)
		self.avg_pool2 = nn.AvgPool2d(3)
		self.conv3 = nn.Conv2d(40, self.final_channels, 11, 1)
		self.avg_pool3 = nn.AvgPool2d(2)
		self.ff1 = nn.Linear(225, self.final_linear)

		self.rnn1 = nn.RNN(vocab_size + self.final_channels *
						   self.final_linear, hidden_size)
		self.ff2 = nn.Linear(hidden_size, vocab_size)
		self.softmax = nn.Softmax(dim=2)

	def forward(self, input, hidden):
		# img.shape=(batch_size, 3, 400, 400), text_batch.shape=(batch_size, seq_len)
		img, text_batch = input
		seq_len, batch_size = text_batch.shape

		# Get Image to a [batch_size, 5000] shape
		img = self.conv1(img)
		img = self.avg_pool1(img)
		img = self.conv2(img)
		img = self.avg_pool2(img)
		img = self.conv3(img)
		img = self.avg_pool3(img)
		img = img.view(-1, 50, 225)
		img = self.ff1(img)
		img = img.view(-1, 5000)

		# One-hot-encode input
		text_batch = F.one_hot(text_batch, num_classes=self.vocab_size).float()

		# match shape of text to be able to concat
		img = img.repeat(seq_len, 1, 1)

		# do rnn and cast to output
		text_out, h_out = self.rnn1(
			torch.cat((img, text_batch), dim=2),
			hidden
		)

		text_out = self.ff2(text_out)
		text_out = self.softmax(text_out)

		return text_out, h_out

	def init_hidden(self, batch_size):
		return torch.zeros(1, batch_size, self.hidden_size, dtype=torch.float32)


def calculate_loss(pred, targ, loss_fxn):
	pred = torch.flatten(pred[:pred.shape[0]-1, :, :], 0, 1)
	targ = torch.flatten(targ[1:, :], 0, 1)
	return loss_fxn(pred, targ)


if __name__ == "__main__":
	dataset = CocoAnnotationDataset()
	loss_fxn = nn.NLLLoss()
	model = ImageCaptioningNet(len(dataset.vocab))
	optimizer = optim.SGD(model.parameters(), lr=0.1)

	# with torch.no_grad():
	# 	img, text = dataset[0]
	# 	print(img.shape, text.shape)
	# 	batch_size = text.shape[1]
	# 	text_out, h_out = model((img, text), model.init_hidden(batch_size))
	# 	loss = calculate_loss(text_out, text, loss_fxn)
	# 	print("loss {}".format(loss))

	for epoch in range(50):
		for (img, text) in CocoAnnotationDataset():
			model.zero_grad() 
			text_out, h_n = model((img, text), model.init_hidden(text.shape[1]))
			loss = calculate_loss(text_out, text, loss_fxn)

			print("Epoch {} Loss {}".format(epoch, loss))

			loss.backward()
			optimizer.step()
