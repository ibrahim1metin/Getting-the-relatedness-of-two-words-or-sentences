import pandas as pd
import numpy as np
import string
import tensorflow_probability as tfp
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, SimpleRNN, RNN,GlobalMaxPool1D,GlobalAveragePooling1D,Input,Embedding,AdditiveAttention,Attention,GRU,MultiHeadAttention,Flatten,LSTM,Conv2D,MaxPool2D,Dropout,ConvLSTM2D,Reshape,Bidirectional,BatchNormalization,TimeDistributed
import tensorflow as tf
MAX_LENGTH=32
MAX=50000
batch_size=64
file='SICK_corrected.tsv'
def clean_text(text):
    wordss=text.split()
    table = str.maketrans('','', string.punctuation)
    wordss=[word.translate(table) for word in wordss]
    ss=[word.lower() for word in wordss if word.isalpha()]
    return ss
data=pd.read_csv(file,sep="\t")
print(data["relatedness_score"])
data["relatedness_score"]=(data["relatedness_score"]-data["relatedness_score"].min())/(data["relatedness_score"].max()-data["relatedness_score"].min())
#data["relatedness_score"]=(data["relatedness_score"]-data["relatedness_score"].mean())/(data["relatedness_score"].std())
"""for i in data["relatedness_score"]:
    print(i)"""
data=pd.concat([data["pair_ID"],data["sentence_A"],data["sentence_B"],data["relatedness_score"],data["SemEval_set"]],axis=1)
uniques=data["SemEval_set"].unique()
train=data[data["SemEval_set"]==uniques[0]]
test=data[data["SemEval_set"]==uniques[2]]
val=data[data["SemEval_set"]==uniques[1]]
train=train.to_numpy()
test=test.to_numpy()
val=val.to_numpy()
sentaTrain=[i[1] for i in train]
sentbTrain=[i[2] for i in train]
sentaTest=[i[1] for i in test]
sentbTest=[i[2] for i in test]
sentaVal=[i[1] for i in val]
sentbVal=[i[2] for i in val]
scoreTrain=np.asarray([i[3] for i in train])
scoreTest=np.asarray([i[3] for i in test])
scoreVal=np.asarray([i[3] for i in val])
all=(sentaTrain+sentbTrain+sentaTest+sentbTest+sentaVal+sentbVal)

for i in range(len(sentaTrain)):
    sentaTrain[i]=clean_text(sentaTrain[i])
    sentbTrain[i]=clean_text(sentbTrain[i])
for i in range(len(sentaTest)):
    sentaTest[i]=clean_text(sentaTest[i])
    sentbTest[i]=clean_text(sentbTest[i])
for i in range(len(sentaVal)):
    sentaVal[i]=clean_text(sentaVal[i])
    sentbVal[i]=clean_text(sentbVal[i])
tokenizer=Tokenizer(50000)
tokenizer.fit_on_texts(sentaTrain+sentaTest+sentaVal+sentbTest+sentbTrain+sentbVal)
numerical_sent_A_Train=tokenizer.texts_to_sequences(sentaTrain)
numerical_sent_B_Train=tokenizer.texts_to_sequences(sentbTrain)
numerical_sent_A_Test=tokenizer.texts_to_sequences(sentaTest)
numerical_sent_B_Test=tokenizer.texts_to_sequences(sentbTest)
numerical_sent_A_Val=tokenizer.texts_to_sequences(sentaVal)
numerical_sent_B_Val=tokenizer.texts_to_sequences(sentbVal)
numerical_sent_A_Train=pad_sequences(numerical_sent_A_Train,MAX_LENGTH)
numerical_sent_B_Train=pad_sequences(numerical_sent_B_Train,MAX_LENGTH)
numerical_sent_A_Test=pad_sequences(numerical_sent_A_Test,MAX_LENGTH)
numerical_sent_B_Test=pad_sequences(numerical_sent_B_Test,MAX_LENGTH)
numerical_sent_A_Val=pad_sequences(numerical_sent_A_Val,MAX_LENGTH)
numerical_sent_B_Val=pad_sequences(numerical_sent_B_Val,MAX_LENGTH)
numerical_sent_A_Test=np.asarray(numerical_sent_A_Test)
numerical_sent_A_Train=np.asarray(numerical_sent_A_Train)
numerical_sent_A_Val=np.asarray(numerical_sent_A_Val)
numerical_sent_B_Test=np.asarray(numerical_sent_B_Test)
numerical_sent_B_Train=np.asarray(numerical_sent_B_Train)
numerical_sent_B_Val=np.asarray(numerical_sent_B_Val)
tester=("man","person")
tester=(tokenizer.texts_to_sequences([tester[0]]),tokenizer.texts_to_sequences([tester[1]]))
tester=(pad_sequences(tester[0],MAX_LENGTH),pad_sequences(tester[1],MAX_LENGTH))
voc_size=len(tokenizer.word_index)+1
input1=Input(shape=(MAX_LENGTH,))
input2=Input(shape=(MAX_LENGTH,))
emb=Embedding(voc_size,64,input_length=MAX_LENGTH)
lstm1=LSTM(64,return_sequences=True)
lstm2=LSTM(32)
dense1=Dense(MAX_LENGTH,activation="relu",kernel_regularizer=tf.keras.regularizers.L1())
emb1=emb(input1)
emb2=emb(input2)
lstm1_1=lstm1(emb1)
lstm1_2=lstm1(emb2)
lstm2_1=lstm2(lstm1_1)
lstm2_2=lstm2(lstm1_2)
siamense1=dense1(lstm2_1)
siamense2=dense1(lstm2_2)
class CosineLayer(tf.keras.layers.Layer):
    def __init__(self,**kwargs):
        super(CosineLayer,self).__init__(dynamic=False)
    def build(self, input_shape):
        self.built=True
    def call(self,i1,i2):
        sim=tf.keras.losses.cosine_similarity(i1,i2,1)
        return sim
result=CosineLayer()(siamense1,siamense2)
result=Flatten()(result)
result=Dropout(0.1)(result)
result=BatchNormalization(axis=-1)(result)
result=Dense(1,kernel_regularizer=tf.keras.regularizers.L2(),activation="sigmoid")(result)
model=tf.keras.Model(inputs=[input1,input2],outputs=result)
model.compile(metrics=["Accuracy"],loss="MeanSquaredError",optimizer=tf.keras.optimizers.experimental.AdamW())
print(model.summary())
print(scoreTrain)
print(numerical_sent_A_Train)
model.fit([numerical_sent_A_Train,numerical_sent_B_Train],scoreTrain,epochs=50,batch_size=batch_size,validation_data=((numerical_sent_A_Val,numerical_sent_B_Val),scoreVal))
model.evaluate((numerical_sent_A_Test,numerical_sent_B_Test),scoreTest,batch_size=batch_size)
print(model.predict(tester,batch_size=1))
