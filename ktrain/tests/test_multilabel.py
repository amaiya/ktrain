#!/usr/bin/env python3
"""
Tests of ktrain text classification flows
"""
import sys
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.insert(0,'../..')
from unittest import TestCase, main, skip
import ktrain

def synthetic_multilabel():


    import numpy as np

    # data
    X = [[1,0,0,0,0,0,0],
          [1,2,0,0,0,0,0],
          [3,0,0,0,0,0,0],
          [3,4,0,0,0,0,0],
          [2,0,0,0,0,0,0],
          [3,0,0,0,0,0,0],
          [4,0,0,0,0,0,0],
          [2,3,0,0,0,0,0],
          [1,2,3,0,0,0,0],
          [1,2,3,4,0,0,0],
          [0,0,0,0,0,0,0],
          [1,1,2,3,0,0,0],
          [2,3,3,4,0,0,0],
          [4,4,1,1,2,0,0],
          [1,2,3,3,3,3,3],
          [2,4,2,4,2,0,0],
          [1,3,3,3,0,0,0],
          [4,4,0,0,0,0,0],
          [3,3,0,0,0,0,0],
          [1,1,4,0,0,0,0]]
                                                                    
    Y = [[1,0,0,0],                                                     
        [1,1,0,0],                                                      
        [0,0,1,0],                                                      
        [0,0,1,1],
        [0,1,0,0],
        [0,0,1,0],
        [0,0,0,1],
        [0,1,1,0],
        [1,1,1,0],
        [1,1,1,1],
        [0,0,0,0],
        [1,1,1,0],
        [0,1,1,1],
        [1,1,0,1],
        [1,1,1,0],
        [0,1,0,0],
        [1,0,1,0],
        [0,0,0,1],
        [0,0,1,0],
        [1,0,0,1]]
        
    # model   
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.layers import Embedding
    from keras.layers import GlobalAveragePooling1D
    import numpy as np

    X = np.array(X)
    Y = np.array(Y)
    MAXLEN = 7
    MAXFEATURES = 4
    NUM_CLASSES = 4
    model = Sequential()
    model.add(Embedding(MAXFEATURES+1,
                        50,
                        input_length=MAXLEN))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #model.fit(X, Y,
              #batch_size=1,
              #epochs=200,
              #validation_data=(X, Y))
    learner = ktrain.get_learner(model, 
                                 train_data=(X, Y),
                                 val_data=(X, Y),
                                 batch_size=1)
    learner.lr_find()
    hist = learner.fit(0.001, 200)
    learner.view_top_losses(n=5)
    learner.validate()
    return hist



class TestMultilabel(TestCase):

    def test_multilabel(self):
        hist  = synthetic_multilabel()
        final_acc = hist.history['val_acc'][-1]
        print('final_accuracy:%s' % (final_acc))
        self.assertGreater(final_acc, 0.97)

if __name__ == "__main__":
    main()
