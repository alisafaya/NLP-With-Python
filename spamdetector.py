# -*- coding: utf-8 -*-
from numpy import genfromtxt
import random

dataset = genfromtxt('data/SpamDataset/spambase.data', delimiter=',')
random.shuffle(dataset)
training_set = dataset[:3000]
test_set = dataset[3000:]
x_train = training_set[:, :57]
y_train = training_set[:, 57]
x_test = test_set[:, :57]
y_test = test_set[:, 57]



