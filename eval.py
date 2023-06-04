from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
import tensorflow as tf
import pandas as pd
import numpy as np

def evaluate_model(model, x_test, y_test):
    '''
    evaluate trained model on test set
    :param model:
    :param x_test:
    :param y_test:
    :return:
    '''
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("Test accuracy", test_acc)
    print("Test loss", test_loss)
    return test_loss, test_acc

def test_model(model, x_test):
    '''
    test trained model on test set
    :param model:
    :param x_test:
    :return:
    '''
    predictions = model.predict(x_test, workers=-1, verbose=1)
    return predictions

def get_classification_report(y_test, predictions, classes, save_cr_path):
    '''
    get classification report
    :param y_test:
    :param predictions:
    :param classes:
    :param save_cr_path:
    :return:
    '''
    cr = classification_report(y_test, predictions.argmax(axis=1), target_names=classes, output_dict=True)
    print(cr)
    repdf = pd.DataFrame(cr).round(2).transpose()
    repdf.insert(loc=0, column='class', value=['Normal IOP', 'High IOP'] + ["accuracy", "macro avg", "weighted avg"])
    repdf.to_csv(save_cr_path, index=False)

def get_trained_model(model_path):
    '''
    load trained model
    :param model_path:
    :return:
    '''
    reconstructed_model = keras.models.load_model(model_path)
    reconstructed_model.summary()
    return reconstructed_model

def get_model_filter_shapes(reconstructed_model):
    '''
    extract trained model's conv layers filters
    :param reconstructed_model:
    :return:
    '''
    # summarize filter shapes
    for idx, layer in enumerate(reconstructed_model.layers):
        # check for convolutional layer
        if 'conv' not in layer.name:
            continue
        # get filter weights
        filters, biases = layer.get_weights()
        print(idx, layer.name, filters.shape)

def get_model_feature_maps(reconstructed_model, x_test):
    '''
    extract trained model's conv layer feature maps
    :param reconstructed_model:
    :param x_test:
    :return:
    '''
    # Sub Model
    global ff
    sub_model = tf.keras.Sequential()
    sub_model.add(reconstructed_model.input)
    sub_model.add(reconstructed_model.layers[1])
    # ixs = [1, 4, 7]
    # for i in ixs:
    #     sub_model.add(reconstructed_model.layers[i])
    print(sub_model.summary())
    feature_maps = sub_model.predict([x_test])
    for i, f in enumerate(feature_maps):
        for ff in f[:20]:
            ff = ff.clip(min=0)
            ff = ff / ff.max()
            ff = interp1d(ff, x_test[0].shape[0])
    print(feature_maps.shape)
    print(feature_maps[0].shape)
    print(ff/max(ff.max(), abs(ff.min())))

def interp1d(array, new_len):
    '''
    interpolate 1d array
    :param array:
    :param new_len:
    :return:
    '''
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)
