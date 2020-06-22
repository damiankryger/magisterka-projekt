# Master's thesis - project

This is a project that was created as part of creating my master's thesis. It is a program that creates new songs using the capabilities of the LSTM network.

## How to run?

1. Please make sure you are using Python version 3.6. To run this project it is also required to configure the local environment so that Tensorflow can use the graphics card. Here you will find an article that will guide you step by step through the process of configuring your environment: [link](https://towardsdatascience.com/how-to-install-tensorflow-gpu-on-ubuntu-18-04-1c1d2d6d6fd2).
2. Create a virtual development environment using the `python3 -m venv .venv` command.
3. Aktywuj wirtualne środowisko za pomocą komendy source `.venv/bin/activate`.
4. Install dependencies that are in the requirements.txt file using the `pip install -r requirements.txt` command
5. To run the neural network learning process, run the `python3 -m src.train.py` command. This step is required before performing the next step.
6. The next step is to start the song generation based on the parameters determined in the previous step. To do this, run the `python -m src.generate.py` command. Zostanie wygenerowany plik z rozszerzeniem .mid, który następnie w celu weryfikacji należy uruchomić w odpowiednim programie (np. LMMS na Ubuntu).
