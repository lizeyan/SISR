class Layer(object):
    def __init__(self, name, trainable=False):
        self.name = name
        self.trainable = trainable

    def forward(self, inputs):
        pass

