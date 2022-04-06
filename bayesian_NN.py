import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Dense, Lambda
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import Sequence


#############################
#       Read data           #
#############################


# Read person, score, and shifts from data
def read_personal_data(person):
    data = pd.read_fwf(person)
    a = []
    for row in range(len(data)):
        r = split_data(data.iloc[row].values)
        a.append(r)

    df_new = pd.DataFrame(a)
    return df_new


# Split personal data into values
def split_data(row):
    arr_row = str(row)
    arr_row = arr_row.replace("[", "")
    arr_row = arr_row.replace("]", "")
    arr_row = arr_row.replace("'", "")
    result = arr_row.split(",")
    return result


# Get dataframes for each person scoring shifts
def get_personal_data(data):
    data, num_people = get_num_people(data)
    dfs = []
    for i in range(num_people):
        df = data.loc[data['Person_category'] == i]
        lst = df.index[df['Score'] == '-1'].tolist()
        df = df.drop(lst)
        dfs.append(df)
    return dfs


# Get number of different people who have answered
def get_num_people(data):
    data.rename(columns={1: 'Person', 0: 'Score'}, inplace=True)
    data['Person'] = data['Person'].astype('category')
    data['Person_category'] = data['Person'].cat.codes

    num_people, cols = data.groupby('Person').count().shape

    return data, num_people


#################################
#       Tutorial on BNN         #
#################################

FEATURE_NAMES = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]
hidden_units = [8, 8]
learning_rate = 0.001


def create_model_inputs():
    inputs = {}
    for feature_name in range(63):
        inputs[feature_name] = layers.Input(
            name=str(feature_name), shape=(1,), dtype=tf.float32
        )
    print("INPUTS: ", inputs)
    return inputs


def get_train_and_test_splits(train_size, batch_size=1):
    # We prefetch with a buffer the same size as the dataset because th dataset
    # is very small and fits into memory.
    dataset_old = (
        tfds.load(name="wine_quality", as_supervised=True, split="train")
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=dataset_size)
        .cache()
    )
    print("DATASET: ", dataset_old)
    '''
    data = tfds.load(name="wine_quality", as_supervised=True, split="train")
    data2 = pd.read_csv('cleaned_df')
    data2 = tf.convert_to_tensor(data2)
    # data2 = tf.data.TextLineDataset(["cleaned_df"])
    print(data, type(data))
    print(data2, type(data2))
    print("DONE")

    '''
    data2 = pd.read_csv('cleaned_df')
    print(data2, data2.shape)
    data2 = data2.iloc[:10000, :]
    print(data2, data2.shape)
    var_target = data2.pop('score')
    new_tf_dataset = tf.data.Dataset.from_tensor_slices((data2.values, var_target.values))
    print(new_tf_dataset)

    dataset = (
        new_tf_dataset
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))
        .prefetch(buffer_size=dataset_size)
        .cache()
    )
    print("New DATASET: ", dataset)

    # We shuffle with a buffer the same size as the dataset.
    train_dataset = (
        dataset.take(train_size).shuffle(buffer_size=train_size).batch(batch_size)
    )
    test_dataset = dataset.skip(train_size).batch(batch_size)
    return train_dataset, test_dataset


def run_experiment(model, loss, train_dataset, test_dataset):
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],
    )
    print("Start training the model...")
    model.fit(train_dataset, epochs=num_epochs, validation_data=test_dataset)
    print("Model training finished.")
    _, rmse = model.evaluate(train_dataset, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")
    print("Evaluating model performance...")
    _, rmse = model.evaluate(test_dataset, verbose=0)
    print(f"Test RMSE: {round(rmse, 3)}")


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(
                tfp.layers.MultivariateNormalTriL.params_size(n), dtype=dtype
            ),
            tfp.layers.MultivariateNormalTriL(n),
        ]
    )
    return posterior_model


def create_bnn_model(train_size):
    inputs = create_model_inputs()
    features = keras.layers.concatenate(list(inputs.values()))
    print("FEATURES 1: ", features)
    features = layers.BatchNormalization()(features)
    print("FEATURES 2: ", features)
    # Create hidden layers with weight uncertainty using the DenseVariational layer.
    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=1 / train_size,
            activation="sigmoid",
        )(features)
    print("FEATURES 3: ", features)
    # The output is deterministic: a single point estimate.
    outputs = layers.Dense(units=1)(features)
    print("OUTPUTS: ", outputs)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


def compute_predictions(model, iterations=100):
    predicted = []
    for _ in range(iterations):
        predicted.append(model(examples).numpy())

    predicted = np.concatenate(predicted, axis=1)
    prediction_mean = np.mean(predicted, axis=1).tolist()
    prediction_min = np.min(predicted, axis=1).tolist()
    prediction_max = np.max(predicted, axis=1).tolist()
    prediction_range = (np.max(predicted, axis=1) - np.min(predicted, axis=1)).tolist()

    for idx in range(sample):
        print(
            f"Predictions mean: {round(prediction_mean[idx], 2)}, "
            f"min: {round(prediction_min[idx], 2)}, "
            f"max: {round(prediction_max[idx], 2)}, "
            f"range: {round(prediction_range[idx], 2)} - "
            f"Actual: {targets[idx]}"
        )


#################################
#       Create Bayesian NN      #
#################################

def gaussian_layer(x):
    '''
    # Get the number of dimensions of the input
    num_dims = len(x.get_shape())

    # Separate the parameters
    mu, sigma = tf.unstack(x, num=2, axis=-1)

    # Add one dimension to make the right shape
    mu = tf.expand_dims(mu, -1)
    sigma = tf.expand_dims(sigma, -1)

    # Apply a softplus to make positive
    mu = tf.keras.activations.sigmoid(mu)
    sigma = tf.keras.activations.sigmoid(sigma)

    # Join back together again
    out_tensor = tf.concat((mu, sigma), axis=num_dims - 1)

    return out_tensor
    '''

    """
    Lambda function for generating normal distribution parameters
    mu (mean) and sigma (variance) from a Dense(2) output.
    Assumes tensorflow 2 backend.

    Usage
    -----
    outputs = Dense(2)(final_layer)
    distribution_outputs = Lambda(negative_binomial_layer)(outputs)

    Parameters
    ----------
    x : tf.Tensor
        output tensor of Dense layer

    Returns
    -------
    out_tensor : tf.Tensor

    """

    # Get the number of dimensions of the input
    num_dims = len(x.get_shape())

    # Separate the parameters
    n, p = tf.unstack(x, num=2, axis=-1)

    # Add one dimension to make the right shape
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)

    # Apply a softplus to make positive
    n = tf.keras.activations.softplus(n)

    # Apply a sigmoid activation to bound between 0 and 1
    p = tf.keras.activations.sigmoid(p)

    # Join back together again
    out_tensor = tf.concat((n, p), axis=num_dims - 1)

    return out_tensor


def negative_binomial_loss(y_true, y_pred):
    """
    Negative binomial loss function.
    Assumes tensorflow backend.

    Parameters
    ----------
    y_true : tf.Tensor
        Ground truth values of predicted variable.
    y_pred : tf.Tensor
        n and p values of predicted distribution.

    Returns
    -------
    nll : tf.Tensor
        Negative log likelihood.
    """

    # Separate the parameters
    n, p = tf.unstack(y_pred, num=2, axis=-1)

    # Add one dimension to make the right shape
    n = tf.expand_dims(n, -1)
    p = tf.expand_dims(p, -1)

    # Calculate the negative log likelihood
    nll = (
            tf.math.lgamma(n)
            + tf.math.lgamma(y_true + 1)
            - tf.math.lgamma(n + y_true)
            - n * tf.math.log(p)
            - y_true * tf.math.log(1 - p)
    )

    '''
    n = len(y_pred)
    val = 2
    n = tf.cast(n, tf.float32)
    tf.cast(sigma, tf.float32)
    val = tf.cast(val, tf.float32)
    tf.cast(math.pi, tf.float32)

    print(type(val), type(math.pi), type(n))

    calc_loss = - (n / val) * tf.math.log(val * math.pi)
                # - (n / val) * tf.math.log(sigma**val) \
                # - (1 / (val*sigma**val))*sum(y_true-mu)
    '''

    return nll


def bayesian_nn(x_train, y_train):
    model = keras.Sequential(name="BayesianPP")

    probability_layer = Lambda(gaussian_layer)

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.add(probability_layer)

    model.compile(loss=negative_binomial_loss, optimizer=Adam(), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, validation_split=0.1, epochs=50, verbose=0)

    print(model.summary())

    return model, history


# Compile and fit pp model that is trained from scratch
def scratch_model(x_train, y_train):
    model = Sequential(name="PersonalPreferenceModel2")

    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.2, momentum=0.9), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, validation_split=0.1, epochs=50, verbose=0)

    print(model.summary())

    return model, history


def prob_NN(X_train, y_train):
    unc = y_train
    inputs = Input(shape=(63, ), name='input')

    x = Dense(256, activation='relu')(inputs)
    x = Dense(128, activation='relu')(x)

    # Head 1 (predicted score)
    x = Dense(32, activation='relu')(x)
    y = Dense(1, activation='sigmoid', name='prediction')(x)

    # Head 2 (error of prediction)
    x = Dense(32, activation='relu')(x)
    z = Dense(1, activation='sigmoid', name='error')(x)

    # uncertainty = CustomCallback(X_train, y_train, unc)
    # y_tr = np.array([y_train, unc])
    # training = CustomTraining(np.array(X_train), np.array(y_train), np.array(unc), 64)

    model = Model(inputs=inputs, outputs=[y, z])
    model.compile(loss={'prediction': 'mean_squared_error', 'error': 'mean_squared_error'},
                  optimizer=SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
                  # loss_weights={'prediction': 0.67, 'error': 0.33},
                  metrics={'prediction': 'mae', 'error': 'mae'})
    training = CustomTraining(np.array(X_train), np.array(y_train), np.array(unc), model, 64)
    history = model.fit(training, epochs=20, verbose=1)    # , callbacks=[uncertainty])

    return model, history


class CustomTraining(Sequence):
    def __init__(self, x_test, y_test, unc, model, batch_size):
        self.x_test = x_test
        self.y_test = y_test
        self.unc = unc
        self.model = model
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x_test) / self.batch_size)

    def __getitem__(self, idx):
        X = self.x_test[idx * self.batch_size:(idx + 1) *
          self.batch_size]
        y = self.y_test[idx * self.batch_size:(idx + 1) *
          self.batch_size]
        u = self.unc[idx * self.batch_size:(idx + 1) *
          self.batch_size]
        return X, y, u

    def on_epoch_end(self):
        y, u = self.model.predict(self.x_test)
        # self.y_test = np.array(self.y_test).reshape((len(self.y_test), 1))
        # print(y_pred[0].shape, self.y_test.shape)
        res = np.absolute(np.subtract(u, y))
        print(type(self.unc), type(y), type(res))
        print(len(self.unc), len(y), len(res))
        print(self.unc.shape, y.shape, res.shape)
        self.unc = res


class CustomCallback(keras.callbacks.Callback):
    def __init__(self, x_test, y_test, unc):
        self.x_test = x_test
        self.y_test = y_test
        self.unc = unc

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x_test)
        self.y_test = np.array(self.y_test).reshape((len(self.y_test), 1))
        # print(y_pred[0].shape, self.y_test.shape)
        res = np.subtract(y_pred[0], self.y_test)
        self.unc = np.absolute(res)


# Check for rows that are exactly the same, remove one of those
def check_same(dataf):
    counter = 0
    same_shifts = []
    data = dataf.values
    for i in range(len(data)):
        for j in range(len(data)):
            if j != i:
                if (data[i, :62] == data[j, :62]).all():
                    counter += 1
                    print("SAME")
                    same = [i, j]
                    same2 = [j, i]
                    if (same not in same_shifts) and (same2 not in same_shifts):
                        same_shifts.append(same)
                        # dataf = dataf.drop([dataf.index[j]])
    return counter, same_shifts, dataf


# Compile and fit pp model (using pre-trained model and re-training it)
def personal_preference_model(x_train, y_train):
    loaded_model = tf.keras.models.load_model("g_scoring_model_new")

    # Freeze some layers
    for layer in loaded_model.layers[:3]:
        layer.trainable = False

    # Check the trainable status of layers
    for layer in loaded_model.layers:
        print(layer, layer.trainable)

    model = Sequential(name="PersonalPreferenceModel")
    model.add(loaded_model)

    model.compile(loss='mse', optimizer=SGD(learning_rate=0.2, momentum=0.9), metrics=MeanAbsoluteError())
    history = model.fit(x_train, y_train, batch_size=64, validation_split=0.1, epochs=50, verbose=0)

    print(model.summary())

    return model, history


# Plot the predictions against real values
def plot_predictions(predicted, actual):
    vals = 21
    figure, ax = plt.subplots(1, 1, figsize=(7, 5))
    major_ticks = np.arange(0, 21, 1)

    ax.plot(predicted[0:vals][0], 'ob', label="predicted1")
    ax.plot(predicted[0:vals][1], 'or', label="predicted2")
    ax.plot(actual.values[0:vals], 'x', label="real")
    ax.set_title("Predicted vs real scores of shifts")
    ax.set_xticks(major_ticks)
    ax.grid(which="major", linestyle='--')
    ax.set_xlim(-1, vals)
    ax.set_ylim(-0.01, 1.01)
    ax.legend()

    plt.show()


if __name__ == '__main__':
    d_size = 10000
    d_s = 2928113
    run_example = False
    kmeans_creation = False

    ###############################
    #      Read general data      #
    ###############################

    df = pd.read_csv('cleaned_df')
    print(df.head(), len(df))

    print(df.iloc[:d_size, :63])
    print(df.iloc[:d_size, 63])

    model_, history_ = prob_NN(df.iloc[:d_size, :63], df.iloc[:d_size, 63])
    preds = model_.predict(df.iloc[d_size:d_size+30, :63])
    act = df.iloc[d_size:d_size+30, 63]

    plot_predictions(preds, act)

    print(type(preds), type(act.values))
    print(preds, act)
    print(preds[0][0])
    print(act.values[0])
    # print(preds[1], actual[1])
    # print(preds[2], actual[2])

    # Read personal data
    p_data = read_personal_data('answers.txt')
    p_dfs = get_personal_data(p_data)
    pers_data = p_dfs[0]  # Choose which personal data to use

    # pers_data = pers_data.iloc[4:74, :]
    print(pers_data)

    # Get shifts and their score
    p_shifts = pers_data.iloc[:, 2:65].apply(pd.to_numeric)
    p_score = np.array(pers_data['Score'].apply(pd.to_numeric))

    # Scale score to a scale 0-1
    p_scaler = MinMaxScaler()
    p_scores = p_scaler.fit_transform(p_score.reshape(-1, 1))

    # Split data into train and test
    pX_train, pX_test, py_train, py_test = train_test_split(p_shifts.values, p_scores, test_size=0.1, shuffle=False)

    '''
    mod, hist = bayesian_nn(pX_train, py_train)
    pred = mod.predict(pX_test)

    mod2, hist2 = scratch_model(pX_train, py_train)
    pred2 = mod2.predict(pX_test)

    print("Predictions Probabilities")
    print(pred, py_test)

    print("Predictions")
    print(pred2, py_test)
    '''

    if kmeans_creation:
        wcss = []
        for i in range(1, 100, 10):
            kmeans = KMeans(i)
            kmeans.fit(df.iloc[:, :63].values)
            wcss_iter = kmeans.inertia_
            wcss.append(wcss_iter)

        number_clusters = range(1, 100, 10)
        plt.plot(number_clusters, wcss)
        plt.title('Elbow test for clustering')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show()

    if run_example:
        dataset_size = 4898
        batch_size = 256
        num_epochs = 100
        sample = 10
        train_size = int(dataset_size * 0.85)
        mse_loss = keras.losses.MeanSquaredError()
        # Split the datasets into two groups
        train_dataset, test_dataset = get_train_and_test_splits(train_size, batch_size)
        # Sample some example targets from the test dataset
        examples, targets = list(test_dataset.unbatch().shuffle(batch_size * 10).batch(sample))[
            0
        ]

        num_epochs = 100
        bnn_model_full = create_bnn_model(train_size)
        run_experiment(bnn_model_full, mse_loss, train_dataset, test_dataset)
        compute_predictions(bnn_model_full)
