import random


class FakeDetector:
    def __init__(self, file):
        """
        :file png
        """
        self.__file = file
        self.__prediction = bool(random.getrandbits(1))

    @property
    def prediction(self):
        return self.__prediction

    @prediction.setter
    def prediction(self, value):
        self.__prediction = value


