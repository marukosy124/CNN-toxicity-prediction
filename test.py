import numpy as np
from tensorflow import keras

pfilename = '../SR-ARE-score/names_onehots.pickle'
# pfilename = '../SR-ARE-test/names_onehots.pickle'
pickle_data = np.load(pfilename, allow_pickle=True)
test_data = pickle_data['onehots']

model = keras.models.load_model('trained_model30.h5')

predictions = model.predict(test_data)

with open("labels.txt", "a+") as f:
    for p in predictions:
        # print(p)
        if p >= 0.5:
            f.write("1\n")
        else:
            f.write("0\n")
