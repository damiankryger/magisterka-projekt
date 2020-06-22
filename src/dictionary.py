import pickle

from music21 import Music21Object


class Dictionary:
    def __init__(self):
        self.value_to_id = {}
        self.id_to_value = {}
        self.counter = 0

    def add(self, value: str):
        key = self.counter

        if self.existsKey(value):
            return self.value_to_id[value]

        self.value_to_id[value] = self.counter
        self.id_to_value[self.counter] = value
        self.counter += 1

        return key

    def existsKey(self, value: str):
        return self.value_to_id.get(value) is not None

    def getSize(self):
        return len(self.value_to_id)

    def dump(self, filename: str):
        with open(filename + '_value_to_id', 'wb') as filepath:
            pickle.dump(self.value_to_id, filepath)

        with open(filename + '_id_to_value', 'wb') as filepath:
            pickle.dump(self.id_to_value, filepath)

    def load(self, filename: str):
        with open(filename + '_value_to_id', 'rb') as filepath:
            self.value_to_id = pickle.load(filepath)

        with open(filename + '_id_to_value', 'rb') as filepath:
            self.id_to_value = pickle.load(filepath)

    def getValue(self, id: int):
        return self.id_to_value[id]


class Sequence():
    def __init__(self):
        self.sequence = []

    def append(self, value: int):
        self.sequence.append(value)

    def getSize(self):
        return len(self.sequence)

    def getOne(self, index: int):
        return self.sequence[index]

    def getList(self, start: int, stop: int):
        return self.sequence[start:stop]

    def dump(self, filename: str):
        with open(filename, 'wb') as filepath:
            pickle.dump(self.sequence, filepath)

    def load(self, filename: str):
        with open(filename, 'rb') as filepath:
            self.sequence = pickle.load(filepath)
