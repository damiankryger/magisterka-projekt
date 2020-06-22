import ast

import numpy
from music21 import note, instrument, chord, stream
from music21.duration import Duration
from numpy.random.mtrand import randint
from tensorflow.python.keras.utils import np_utils

from src.dictionary import Dictionary, Sequence
from src.network import Network

WINDOW_LENGTH = 100

notes_dictionary = Dictionary()
notes_sequence = Sequence()

durations_dictionary = Dictionary()
durations_sequence = Sequence()

network = Network()

# This function is loading data generated in the previous step
def load_data():
    notes_dictionary.load('data/notes_dictionary')
    notes_sequence.load('data/notes_sequence')

    durations_dictionary.load('data/durations_dictionary')
    durations_sequence.load('data/durations_sequence')

# This function divides one large sequence into many smaller ones of a certain length
def prepare_train_sets(sequence):
    train_input = []
    train_output = []

    for index in range(0, sequence.getSize() - WINDOW_LENGTH):
        train_input.append(sequence.getList(index, index + WINDOW_LENGTH))
        train_output.append(sequence.getOne(index + WINDOW_LENGTH))

    patterns_amount = len(train_input)

    train_input = numpy.reshape(train_input, (patterns_amount, WINDOW_LENGTH, 1))

    train_output = np_utils.to_categorical(train_output)

    return (train_input, train_output)

# This function is responsible for creating the sequence of a new song
def predict(dictionary):
    # We draw a sequence based on which a new song will be created
    index = int(randint(0, len(train_input) - 1))

    pattern = train_input[index]
    result = []

    # We create a sequence with a length of 32 elements
    for i in range(32):
        input = numpy.reshape(pattern, (1, len(pattern), 1))
        input = input / dictionary.getSize()

        prediction = network.predict(input)

        index = numpy.argmax(prediction)
        result.append(dictionary.getValue(index))

        pattern = numpy.append(pattern, [index])
        pattern = pattern[1:len(pattern)]

    return result

# This function creates a new song based on the received sequence and data saved in the dictionary
def create_midi(notes, durations):
    offset = 0
    output_notes = []

    for index, pattern in enumerate(notes):
        pattern = ast.literal_eval(pattern)
        duration = ast.literal_eval(durations[index])

        pitch = pattern['pitch']
        duration = float(duration['duration'])

        if isinstance(pitch, list):
            notes_in_chord = []

            for current_note in pitch:
                new_note = note.Note(current_note)
                new_note.storedInstrument = instrument.Piano()
                notes_in_chord.append(new_note)

            new_chord = chord.Chord(notes_in_chord)
            new_chord.offset = offset
            new_chord.duration = Duration(quarterLength=duration)
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pitch)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            new_note.duration = Duration(quarterLength=duration)
            output_notes.append(new_note)

        offset += duration

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp='test_output.mid')


load_data()

# Notes generating
train_input, train_output = prepare_train_sets(notes_sequence)

network.create((train_input.shape[1], train_input.shape[2]), notes_dictionary.getSize())
network.load('models/weights-notes.hdf5')

notes = predict(notes_dictionary)

# Durations generating
train_input, train_output = prepare_train_sets(durations_sequence)

network.create((train_input.shape[1], train_input.shape[2]), durations_dictionary.getSize())
network.load('models/weights-durations.hdf5')

durations = predict(durations_dictionary)

create_midi(notes, durations)
