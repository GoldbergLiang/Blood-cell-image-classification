import os
import cv2
from keras.applications.densenet import DenseNet121
from keras.callbacks import ModelCheckpoint
from keras.engine import Input
from keras.layers import GlobalAveragePooling2D, Dense, Reshape, Lambda, K, LSTM,concatenate
from keras.models import Model
from keras.optimizers import Adam
import numpy as np
from tqdm import tqdm
from keras.utils.np_utils import to_categorical
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import mlxtend.plotting

img_width = 224
img_height = 224
np.random.seed(1337)


def rgb_to_grayscale(input):
    """Average out each pixel across its 3 RGB layers resulting in a grayscale image"""
    return K.mean(input, axis=3)

def rgb_to_grayscale_output_shape(input_shape):
    return input_shape[:-1]

# Loading dataset
print("Dataset loaded")
def get_data(folder):
    """
    Load the data and labels from the given folder.
    """
    X = []
    y = []
    for wbc_type in os.listdir(folder):
        if not wbc_type.startswith('.'):
            if wbc_type in ['NEUTROPHIL']:
                label = 0
            elif wbc_type in ['EOSINOPHIL']:
                label = 1
            elif wbc_type in ['MONOCYTE']:
                label = 2
            elif wbc_type in ['LYMPHOCYTE']:
                label = 3
            else:
                label = 4
            for image_filename in tqdm(os.listdir(folder + wbc_type)):
                img = cv2.imread(folder + wbc_type + '/' + image_filename)
                if img is not None:
                    img = cv2.resize(img, dsize=(224,224), interpolation=cv2.INTER_CUBIC)
                    img_arr = np.asarray(img)
                    X.append(img_arr)
                    y.append(label)
    X = np.asarray(X)
    y = np.asarray(y)
    return X,y

X_train, y_train = get_data('D:/TEST/blood cell image classification/data/TRAIN/')
X_test, y_test = get_data('D:/TEST/blood cell image classification/data/TEST/')


print(y_train.shape,'\n',X_test.shape)
print(y_test.shape,'\n',y_test.shape)
# Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_trainHot = to_categorical(y_train, num_classes = 4)
y_testHot = to_categorical(y_test, num_classes = 4)
dict_characters = {1:'NEUTROPHIL', 2:'EOSINOPHIL', 3:'MONOCYTE', 4:'LYMPHOCYTE'}
print(dict_characters)

# Helper Functions  Learning Curves and Confusion Matrix

def plotKerasLearningCurve():
    plt.figure(figsize=(10,5))
    metrics = np.load('logs.npy')[()]
    filt = ['acc'] # try to add 'loss' to see the loss learning curve
    for k in filter(lambda x : np.any([kk in x for kk in filt]), metrics.keys()):
        l = np.array(metrics[k])
        plt.plot(l, c= 'r' if 'val' not in k else 'b', label='val' if 'val' in k else 'train')
        x = np.argmin(l) if 'loss' in k else np.argmax(l)
        y = l[x]
        plt.scatter(x,y, lw=0, alpha=0.25, s=100, c='r' if 'val' not in k else 'b')
        plt.text(x, y, '{} = {:.4f}'.format(x,y), size='15', color= 'r' if 'val' not in k else 'b')
    plt.legend(loc=4)
    plt.axis([0, None, None, None])
    plt.grid()
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
#This function prints and plots the confusion matrix.
#Normalization can be applied by setting `normalize=True`.

    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_learning_curve(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./accuracy_curve.png')
    #plt.clf()
    # summarize history for loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('./loss_curve.png')



print("Building model...")
input_tensor = Input(shape=(img_width, img_height, 3))

# Creating CNN
cnn_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=input_tensor)

x = cnn_model.output
cnn_bottleneck = GlobalAveragePooling2D()(x)

# Make CNN layers not trainable
for layer in cnn_model.layers:
    layer.trainable = False

# Creating RNN
x = Lambda(rgb_to_grayscale, rgb_to_grayscale_output_shape)(input_tensor)
x = Reshape((32, 1568))(x)  # 32 timesteps, input dim of each timestep 1568
x = LSTM(2048, return_sequences=True)(x)
rnn_output = LSTM(2048)(x)

# Merging both cnn bottleneck and rnn's output wise element wise multiplication
x = concatenate([cnn_bottleneck, rnn_output])
predictions = Dense(4, activation='softmax')(x)
model = Model(input=input_tensor, output=predictions)

print("Model built")

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print("Starting training")
checkpointer1 = ModelCheckpoint(filepath="initial-weights.hdf5", verbose=1, save_best_only=True, mode='max',monitor= 'val_acc')
model.fit(X_train, y_trainHot, epochs = 1, verbose=1,validation_data=(X_test, y_testHot),batch_size = 32,callbacks=[checkpointer1])


print("Initial training done, starting phase two (finetuning)")

# Load two new generator with smaller batch size, needed because using the same batch size
# for the fine tuning will result in GPU running out of memory and tensorflow raising an error
# Load best weights from initial training
model.load_weights("initial-weights.hdf5")

# Make all layers trainable for finetuning
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
              metrics=['accuracy'])

checkpointer2 = ModelCheckpoint(filepath="finetuned-weights.hdf5", verbose=1, save_best_only=True,
                               monitor='val_acc',mode='max')
history=model.fit(X_train, y_trainHot, epochs = 30, verbose=1, validation_data=(X_test, y_testHot),batch_size = 16, callbacks=[checkpointer2])

# Final evaluation of the model
print("Training done, doing final evaluation...")

#load weights
model.load_weights("finetuned-weights.hdf5")

model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',metrics=['accuracy'])

scores = model.evaluate(X_test, y_testHot)
print(model.metrics_names, scores)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

#plot confusion_matrix
pred = model.predict(X_test)
pred = np.argmax(pred,axis = 1)
y_true = np.argmax(y_testHot,axis = 1)
CM = confusion_matrix(y_true, pred)
fig, ax = mlxtend.plotting.plot_confusion_matrix(conf_mat=CM, figsize=(5, 5))
plt.show()

