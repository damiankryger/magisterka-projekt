from glob import glob

import numpy
from music21 import converter
from music21.chord import Chord
from music21.note import Note
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils import np_utils

from src.dictionary import Dictionary, Sequence
from src.network import Network

# Defining constants
FILES = glob('../midi/edm_pack/future_house/*.mid')
WINDOW_LENGTH = 100

# Creating instances of sequences and dictionaries
notes_dictionary = Dictionary()
notes_sequence = Sequence()

durations_dictionary = Dictionary()
durations_sequence = Sequence()

# Creating instance of Network class
network = Network()

# This function is responsible for creating the sequence and dictionary based on processed songs
def build_sequences_and_dictionaries():
    for file in FILES:
        # Parsing MIDI file to flat notes
        elements = converter.parse(file).flat.notes

        # For each note or chord in the elements list
        for element in elements:
            note = None

            if isinstance(element, Note):
                note = {'pitch': str(element.pitch)}

            elif isinstance(element, Chord):
                result = list(map(lambda note: str(note.pitch), element.notes))

                note = {'pitch': result}

            duration = {'duration': str(element.duration.quarterLength)}

            # We can add note and duration to the dictionary only when both values are specified
            if note is not None and duration is not None:
                key = notes_dictionary.add(str(note))
                notes_sequence.append(key)

                key = durations_dictionary.add(str(duration))
                durations_sequence.append(key)

    # Dumping generated data to use them in next step
    notes_dictionary.dump('data/notes_dictionary')
    notes_sequence.dump('data/notes_sequence')
    durations_dictionary.dump('data/durations_dictionary')
    durations_sequence.dump('data/durations_sequence')


# This function divides one large sequence into many smaller ones of a certain length
def prepare_train_sets(dictionary, sequence):
    train_input = []
    train_output = []

    for index in range(0, sequence.getSize() - WINDOW_LENGTH):
        train_input.append(sequence.getList(index, index + WINDOW_LENGTH))
        train_output.append(sequence.getOne(index + WINDOW_LENGTH))

    patterns_amount = len(train_input)

    train_input = numpy.reshape(train_input, (patterns_amount, WINDOW_LENGTH, 1))
    train_input = train_input / float(dictionary.getSize())

    train_output = np_utils.to_categorical(train_output)

    return (train_input, train_output)


build_sequences_and_dictionaries()

# Training network using notes
train_input, train_output = prepare_train_sets(notes_dictionary, notes_sequence)

network.create((train_input.shape[1], train_input.shape[2]), notes_dictionary.getSize())
network.add_callback(ModelCheckpoint(
    "models/weights-notes.hdf5",
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
))
network.train(train_input, train_output)

# Training network using durations
train_input, train_output = prepare_train_sets(durations_dictionary, durations_sequence)

network.create((train_input.shape[1], train_input.shape[2]), durations_dictionary.getSize())
network.add_callback(ModelCheckpoint(
    "models/weights-durations.hdf5",
    monitor='loss',
    verbose=0,
    save_best_only=True,
    mode='min'
))
network.train(train_input, train_output)
