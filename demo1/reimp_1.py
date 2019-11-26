#%%

import json
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import numpy as np
from collections import Counter
import pickle
# %matplotlib inline



USE_CUDA = torch.cuda.is_available()

print(USE_CUDA)


####

def prepare_sequence(seq, to_ix):
    idxs = list(map(lambda w: to_ix[w] if w in to_ix.keys() else to_ix["<UNK>"], seq))
    tensor = Variable(torch.LongTensor(idxs)).cuda() if USE_CUDA else Variable(torch.LongTensor(idxs))
    return tensor


flatten = lambda l: [item for sublist in l for item in sublist]


####

# Data load and Preprocessing

#%%

train = open("/Users/wendizhou/PycharmProjects/intention_classfication/RNN-for-Joint-NLU-master/data/atis-2.train.w-intent.iob","r").readlines()
train = [t[:-1] for t in train]
train = [[t.split("\t")[0].split(" "),t.split("\t")[1].split(" ")[:-1],t.split("\t")[1].split(" ")[-1]] for t in train]
train = [[t[0][1:-1],t[1][1:],t[2]] for t in train]


seq_in,seq_out, intent = list(zip(*train))

#%%

vocab = set(flatten(seq_in))
slot_tag = set(flatten(seq_out))
intent_tag = set(intent)



LENGTH=50

#%%

sin=[]
sout=[]

# %%

for i in range(len(seq_in)):
    temp = seq_in[i]
    if len(temp) < LENGTH:
        temp.append('<EOS>')
        while len(temp) < LENGTH:
            temp.append('<PAD>')
    else:
        temp = temp[:LENGTH]
        temp[-1] = '<EOS>'
    sin.append(temp)

    temp = seq_out[i]
    if len(temp) < LENGTH:
        while len(temp) < LENGTH:
            temp.append('<PAD>')
    else:
        temp = temp[:LENGTH]
        temp[-1] = '<EOS>'
    sout.append(temp)

# %%

word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
for token in vocab:
    if token not in word2index.keys():
        word2index[token] = len(word2index)

index2word = {v: k for k, v in word2index.items()}

tag2index = {'<PAD>': 0}
for tag in slot_tag:
    if tag not in tag2index.keys():
        tag2index[tag] = len(tag2index)
index2tag = {v: k for k, v in tag2index.items()}

intent2index = {}
for ii in intent_tag:
    if ii not in intent2index.keys():
        intent2index[ii] = len(intent2index)
index2intent = {v: k for k, v in intent2index.items()}

# %%

train = list(zip(sin, sout, intent))

# %%

train[0][2]

# %%

train_data = []

for tr in train:
    temp = prepare_sequence(tr[0], word2index)
    temp = temp.view(1, -1)

    temp2 = prepare_sequence(tr[1], tag2index)
    temp2 = temp2.view(1, -1)

    temp3 = Variable(torch.LongTensor([intent2index[tr[2]]])).cuda() if USE_CUDA else Variable(
        torch.LongTensor([intent2index[tr[2]]]))

    train_data.append((temp, temp2, temp3))


# %%

def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp

        yield batch


# %% md

# Modeling

# %%

class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, batch_size=16, n_layers=1):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.batch_size = batch_size

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers, batch_first=True, bidirectional=True)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # self.lstm.weight.data.

    def init_hidden(self, input):
        hidden = Variable(
            torch.zeros(self.n_layers * 2, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(
            torch.zeros(self.n_layers * 2, input.size(0), self.hidden_size))
        context = Variable(
            torch.zeros(self.n_layers * 2, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(
            torch.zeros(self.n_layers * 2, input.size(0), self.hidden_size))
        return (hidden, context)

    def forward(self, input, input_masking):
        """
        input : B,T (LongTensor)
        input_masking : B,T (PAD 마스킹한 ByteTensor)

        <PAD> 제외한 리얼 Context를 다시 만들어서 아웃풋으로
        """

        self.hidden = self.init_hidden(input)

        embedded = self.embedding(input)
        output, self.hidden = self.lstm(embedded, self.hidden)

        real_context = []

        for i, o in enumerate(output):  # B,T,D
            real_length = input_masking[i].data.tolist().count(0)  # 실제 길이
            real_context.append(o[real_length - 1])

        return output, torch.cat(real_context).view(input.size(0), -1).unsqueeze(1)


# %%

class Decoder(nn.Module):

    def __init__(self, slot_size, intent_size, embedding_size, hidden_size, batch_size=16, n_layers=1, dropout_p=0.1):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.embedding_size = embedding_size
        self.batch_size = batch_size

        # Define the layers
        self.embedding = nn.Embedding(self.slot_size, self.embedding_size)  # TODO encoder와 공유하도록 하고 학습되지 않게..

        # self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.embedding_size + self.hidden_size * 2, self.hidden_size, self.n_layers,
                            batch_first=True)
        self.attn = nn.Linear(self.hidden_size, self.hidden_size)  # Attention
        self.slot_out = nn.Linear(self.hidden_size * 2, self.slot_size)
        self.intent_out = nn.Linear(self.hidden_size * 2, self.intent_size)

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        # self.out.bias.data.fill_(0)
        # self.out.weight.data.uniform_(-0.1, 0.1)
        # self.lstm.weight.data.

    def Attention(self, hidden, encoder_outputs, encoder_maskings):
        """
        hidden : 1,B,D
        encoder_outputs : B,T,D
        encoder_maskings : B,T # ByteTensor
        """

        hidden = hidden.squeeze(0).unsqueeze(2)  # 히든 : (1,배치,차원) -> (배치,차원,1)

        batch_size = encoder_outputs.size(0)  # B
        max_len = encoder_outputs.size(1)  # T
        energies = self.attn(encoder_outputs.contiguous().view(batch_size * max_len, -1))  # B*T,D -> B*T,D
        energies = energies.view(batch_size, max_len, -1)  # B,T,D (배치,타임,차원)
        attn_energies = energies.bmm(hidden).transpose(1, 2)  # B,T,D * B,D,1 --> B,1,T
        attn_energies = attn_energies.squeeze(1).masked_fill(encoder_maskings, -1e12)  # PAD masking

        alpha = F.softmax(attn_energies)  # B,T
        alpha = alpha.unsqueeze(1)  # B,1,T
        context = alpha.bmm(encoder_outputs)  # B,1,T * B,T,D => B,1,D

        return context  # B,1,D

    def init_hidden(self, input):
        hidden = Variable(
            torch.zeros(self.n_layers * 1, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(
            torch.zeros(self.n_layers * 2, input.size(0), self.hidden_size))
        context = Variable(
            torch.zeros(self.n_layers * 1, input.size(0), self.hidden_size)).cuda() if USE_CUDA else Variable(
            torch.zeros(self.n_layers * 2, input.size(0), self.hidden_size))
        return (hidden, context)

    def forward(self, input, context, encoder_outputs, encoder_maskings, training=True):
        """
        input : B,L(length)
        enc_context : B,1,D
        """
        # Get the embedding of the current input word
        embedded = self.embedding(input)
        hidden = self.init_hidden(input)
        decode = []
        aligns = encoder_outputs.transpose(0, 1)
        length = encoder_outputs.size(1)
        for i in range(length):  # Input_sequence와 Output_sequence의 길이가 같기 때문..
            aligned = aligns[i].unsqueeze(1)  # B,1,D
            _, hidden = self.lstm(torch.cat((embedded, context, aligned), 2),
                                  hidden)  # input, context, aligned encoder hidden, hidden

            # for Intent Detection
            if i == 0:
                intent_hidden = hidden[0].clone()
                intent_context = self.Attention(intent_hidden, encoder_outputs, encoder_maskings)
                concated = torch.cat((intent_hidden, intent_context.transpose(0, 1)), 2)  # 1,B,D
                intent_score = self.intent_out(concated.squeeze(0))  # B,D

            concated = torch.cat((hidden[0], context.transpose(0, 1)), 2)
            score = self.slot_out(concated.squeeze(0))
            softmaxed = F.log_softmax(score)
            decode.append(softmaxed)
            _, input = torch.max(softmaxed, 1)
            embedded = self.embedding(input.unsqueeze(1))

            # 그 다음 Context Vector를 Attention으로 계산
            context = self.Attention(hidden[0], encoder_outputs, encoder_maskings)
            # 요고 주의! time-step을 column-wise concat한 후, reshape!!
        slot_scores = torch.cat(decode, 1)
        return slot_scores.view(input.size(0) * length, -1), intent_score


# %% md

# Training

# %%

LEARNING_RATE = 0.001
EMBEDDING_SIZE = 64
HIDDEN_SIZE = 64
BATCH_SIZE = 16
LENGTH = 50
STEP_SIZE = 10

# %%

encoder = Encoder(len(word2index), EMBEDDING_SIZE, HIDDEN_SIZE)
decoder = Decoder(len(tag2index), len(intent2index), len(tag2index) // 3, HIDDEN_SIZE * 2)
if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

encoder.init_weights()
decoder.init_weights()

loss_function_1 = nn.CrossEntropyLoss(ignore_index=0)
loss_function_2 = nn.CrossEntropyLoss()
enc_optim = optim.Adam(encoder.parameters(), lr=LEARNING_RATE)
dec_optim = optim.Adam(decoder.parameters(), lr=LEARNING_RATE)

# %%

for step in range(STEP_SIZE):
    losses = []
    for i, batch in enumerate(getBatch(BATCH_SIZE, train_data)):
        x, y_1, y_2 = zip(*batch)
        x = torch.cat(x)
        tag_target = torch.cat(y_1)
        intent_target = torch.cat(y_2)
        x_mask = torch.cat([Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA else Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in x]).view(BATCH_SIZE, -1)
        y_1_mask = torch.cat([Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))).cuda() if USE_CUDA else Variable(
            torch.ByteTensor(tuple(map(lambda s: s == 0, t.data)))) for t in tag_target]).view(BATCH_SIZE, -1)

        encoder.zero_grad()
        decoder.zero_grad()

        output, hidden_c = encoder(x, x_mask)
        start_decode = Variable(torch.LongTensor([[word2index['<SOS>']] * BATCH_SIZE])).cuda().transpose(1,
                                                                                                         0) if USE_CUDA else Variable(
            torch.LongTensor([[word2index['<SOS>']] * BATCH_SIZE])).transpose(1, 0)

        tag_score, intent_score = decoder(start_decode, hidden_c, output, x_mask)

        loss_1 = loss_function_1(tag_score, tag_target.view(-1))
        loss_2 = loss_function_2(intent_score, intent_target)

        loss = loss_1 + loss_2
        losses.append(loss.data.cpu().numpy()[0] if USE_CUDA else loss.data.numpy()[0])
        loss.backward()

        torch.nn.utils.clip_grad_norm(encoder.parameters(), 5.0)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), 5.0)

        enc_optim.step()
        dec_optim.step()

        if i % 100 == 0:
            print("Step", step, " epoch", i, " : ", np.mean(losses))
            losses = []

# %% md

# Test

# %%

from data import *
from model import Encoder, Decoder

# %%

_, word2index, tag2index, intent2index = preprocessing('../dataset/corpus/atis-2.train.w-intent.iob', 60)

# %%

index2tag = {v: k for k, v in tag2index.items()}
index2intent = {v: k for k, v in intent2index.items()}

# %%

encoder = Encoder(len(word2index), 64, 64)
decoder = Decoder(len(tag2index), len(intent2index), len(tag2index) // 3, 64 * 2)

encoder.load_state_dict(torch.load('models/jointnlu-encoder.pkl'))
decoder.load_state_dict(torch.load('models/jointnlu-decoder.pkl'))
if USE_CUDA:
    encoder = encoder.cuda()
    decoder = decoder.cuda()

# %%

test = open("../dataset/corpus/atis-2.dev.w-intent.iob", "r").readlines()
test = [t[:-1] for t in test]
test = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in test]
test = [[t[0][1:-1], t[1][1:], t[2]] for t in test]

# %%

index = random.choice(range(len(test)))
test_raw = test[index][0]
test_in = prepare_sequence(test_raw, word2index)
test_mask = Variable(torch.ByteTensor(tuple(map(lambda s: s == 0, test_in.data)))).cuda() if USE_CUDA else Variable(
    torch.ByteTensor(tuple(map(lambda s: s == 0, test_in.data)))).view(1, -1)
start_decode = Variable(torch.LongTensor([[word2index['<SOS>']] * 1])).cuda().transpose(1, 0) if USE_CUDA else Variable(
    torch.LongTensor([[word2index['<SOS>']] * 1])).transpose(1, 0)

output, hidden_c = encoder(test_in.unsqueeze(0), test_mask.unsqueeze(0))
tag_score, intent_score = decoder(start_decode, hidden_c, output, test_mask)

v, i = torch.max(tag_score, 1)
print("Input Sentence : ", *test[index][0])
print("Truth        : ", *test[index][1])
print("Prediction : ", *list(map(lambda ii: index2tag[ii], i.data.tolist())))
v, i = torch.max(intent_score, 1)
print("Truth        : ", test[index][2])
print("Prediction : ", index2intent[i.data.tolist()[0]])
