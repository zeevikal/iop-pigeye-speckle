from tensorflow import keras

def make_model(input_shape, num_classes):
    '''
    model's architecture
    :param input_shape:
    :param num_classes:
    :return:
    '''
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=64, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(num_classes, activation="softmax")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def create_model(x_train):
    '''
    prep model
    :param x_train:
    :return:
    '''
    return make_model(input_shape=x_train.shape[1:])

def set_callbacks(trained_model_path):
    '''
    set callbacks for training process
    :param trained_model_path:
    :return:
    '''
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            trained_model_path, save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=20, min_lr=0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, verbose=1),
    ]
    return callbacks

def train(trained_model_path, model, x_train, y_train, epochs=200, batch_size=64):
    '''
    run training process
    :param trained_model_path:
    :param model:
    :param x_train:
    :param y_train:
    :param epochs:
    :param batch_size:
    :return:
    '''
    # epochs = 500
    # batch_size = 32
    callbacks = set_callbacks(trained_model_path)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=callbacks,
        validation_split=0.2,
        verbose=1,
    )
    return history
