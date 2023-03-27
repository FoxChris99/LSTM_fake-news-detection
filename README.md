# LSTM fake-news-detection
Simple flask web application for fake news detection that uses a Long Short-Term Memory (LSTM) neural network trained with TensorFlow Keras.

| Layers              |       units/params     |
|---------------------|:----------------------:|
| Input               |                        |
| Embedding           |      50.0000/100/250   |
| SpatialDropout1D    |           0.2          |
| LSTM layer          |           100          | 
| Output(sigmoid)     |            1           |

The input text is preprocessed: padded to a lenght of 250 words per sequence, converted to lower-case and stop-words are removed.  Then the text is encoded into a sequence of integers with the Tokenizer, based on the 50.0000 most frequent words in the training set.

The embedding layer takes as input the encoded sequences and converts them into continous vectors of fixed size equal to 100.
The [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory) layer, preceded by [SpatialDropout](https://keras.io/api/layers/regularization_layers/spatial_dropout1d/), is a type of recurrent neural network that is able to capture long-term dependencies in sequential data.
In the end the samples are binary classified with a sigmoid activation function.

![](static/Lstm_unit.png)

