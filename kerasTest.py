import tensorflow as tf
import keras as keras

vision_model = keras.applications.ResNet50()

#video encoding
video_input = keras.Input(shape=(100, None, None, 3))
encoded_frame_sequence = keras.layers.TimeDistributed(vision_model)(video_input)
encoded_video = keras.layers.LSTM(256)(encoded_frame_sequence)

#text-processing
question_input = keras.Input(shape=(100,), dtype='int32')
embedded_question = keras.layers.Embedding(10000, 256)(question_input)
encoded_question = keras.layers.LSTM(256)(embedded_question)

#video question answering
merged = keras.layers.concatenate([encoded_video, encoded_question])
output = keras.layesr.Dense(1000, activation='softmax')(merged)
video_qa_model = keras.Model(inputs=[video_input, question_input],outputs=output)

print('/n/nDone')