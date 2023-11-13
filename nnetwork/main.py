import random


class FakeDetector:
    def __init__(self, file):
        self.__file = file
        self.__prediction = bool(random.getrandbits(1))

    @property
    def prediction(self):
        return self.__prediction

    @prediction.setter
    def name(self, value):
        self.__prediction = value
