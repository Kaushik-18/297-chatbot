# 297-chatbot

### Running the code
Assumptions: python3 and pipenv are installed.

### Dependencies to run the code
- TensorFlow
- Numpy
- Nltk
- Flask
- Gunicorn
- Tflearn

A Pipfile with the dependencies is present in the model folder. Pipenv can be
used to install the dependencies

### Grammar files
The grammar files are present in the grammar folder. The
DeterministicGenerator.py from the JSGF Tools
(https://github.com/syntactic/JSGFTools) can be used to generate the datasets
from the grammar files.

### Training the model
The following command can be used to train the DNN model
pipenv run python chatbot-trainer.py

### Run the model on localhost
The following command can be used to run the Flask API
gunicorn app:app -b localhost:8765

### Test with a query
curl  -H "Content-Type: application/json" -X POST -d '{"message":"who are
you?"}' http://localhost:8765
