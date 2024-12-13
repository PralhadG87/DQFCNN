import tensorflow as tf
import numpy as np
from Proposed_DQFCNN import qnn_gen as qg
from sklearn.model_selection import train_test_split
from Main import fusion



def qdnn_classify(data,l2,label,itr,tr,Acc,TPR,FPR):
    Fea,Label=fusion.callmain(data,l2,label)

    x_train, x_test, y_train, y_test = train_test_split(Fea, Label, train_size=tr,
                                                        random_state=0)

    n_qubits = 6                  # Number of qubits
    num_layers = 8                # Number of layers




    is_data_reduced = True        # Data is reduced to n classes
    reduced_classes = [1,2,3,7]   # Selected (and sorted) classes

    # Number of reduced classes
    reduced_num_classes = len(reduced_classes)
    x_train =np.concatenate((x_train,x_train),axis=1)

    x_test = np.concatenate((x_test, x_test), axis=1)


    # All indexes
    def quantum_network():
        train_index_f = (y_train == -1)
        tests_index_f = (y_test  == -1)

        if is_data_reduced:
          # Filter indexes
          for n_class in reduced_classes:
            train_index_f   |= (y_train == n_class)
            tests_index_f   |= (y_test == n_class)

        num_classes_q = reduced_num_classes

        # New databases
        X_ends_pre = x_train[train_index_f]
        Y_ends_pre = y_train[train_index_f]

        X_tests_pre = x_test[tests_index_f]
        Y_tests_pre = y_test[tests_index_f]


        if is_data_reduced:
          # Change categories to their new range.
          # E.g. {0,...,9} -> {0,...,4}
          for i, k in enumerate(reduced_classes):
            Y_ends_pre[Y_ends_pre == k] = i
            Y_tests_pre[Y_tests_pre == k] = i

        latent_dim = 2 ** n_qubits    # Selected latent dimensions

        class qdnncoder(tf.keras.models.Model):
          def __init__(self, latent_dim):
            super(qdnncoder, self).__init__()
            self.latent_dim = latent_dim
            self.encoder = tf.keras.Sequential([
              tf.keras.layers.Flatten(name = "faltten_1"),
              tf.keras.layers.Dense(10, activation='relu', name = "dense_1"),
              tf.keras.layers.Dense(64, activation='relu', name = "dense_2"),
              tf.keras.layers.Dense(latent_dim, activation='sigmoid', name = "dense_3"),
            ])
            self.decoder = tf.keras.Sequential([
              tf.keras.layers.Dense(64, activation='relu', name = "dense_4"),
              tf.keras.layers.Dense(196, activation='relu', name = "dense_5"),
              tf.keras.layers.Dense(784, activation='relu', name = "dense_6"),
              tf.keras.layers.Reshape((28, 28), name = "reshape_1")
            ])

          def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

        # Prepare and compile the model
        autoencoder = qdnncoder(latent_dim)
        autoencoder.compile(optimizer='adam', loss='mae', metrics=["accuracy"])

        # Train the model with the filtered data
        autoencoder.fit(X_ends_pre, X_ends_pre, epochs=itr, shuffle=True, validation_data=(X_tests_pre, X_tests_pre))

        # Encode data with our new autoencoder
        QX_train = autoencoder.encoder(X_ends_pre).numpy()
        QX_test = autoencoder.encoder(X_tests_pre).numpy()

        # Change Y values to categorical
        QY_train = tf.keras.utils.to_categorical(Y_ends_pre, num_classes_q)
        QY_test = tf.keras.utils.to_categorical(Y_tests_pre, num_classes_q)

        import pennylane as qml
        from pennylane import numpy as p_np

        from pennylane.templates.state_preparations import MottonenStatePreparation
        from pennylane.templates.layers import StronglyEntanglingLayers

        dev = qml.device("default.qubit", wires=n_qubits)


        @qml.qnode(dev, diff_method='adjoint')
        def circuit(weights, inputs=None):
            ''' Quantum QVC Circuit'''

            # Splits need to be done through the tensorflow interface
            weights_each_layer = tf.split(weights, num_or_size_splits=num_layers, axis=0)

            # Input normalization
            inputs_1 = inputs / p_np.sqrt(max(p_np.sum(inputs ** 2, axis=-1), 0.001))

            for i, W in enumerate(weights):
                # Data re-uploading technique
                if i % 2 == 0:
                    MottonenStatePreparation(inputs_1, wires=range(n_qubits))

                # Neural network layer
                StronglyEntanglingLayers(weights_each_layer[i], wires=range(n_qubits))

            # Measurement return
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        weight_shapes = {"weights": (num_layers,n_qubits,3)}

        # Model
        input_m = tf.keras.layers.Input(shape=(2 ** n_qubits,), name = "input_0")
        keras_1 = qml.qnn.KerasLayer(circuit, weight_shapes, output_dim=n_qubits, name = "keras_1")(input_m)
        output = tf.keras.layers.Dense(num_classes_q, activation='softmax', name = "dense_1")(keras_1)

        # Model creation
        model = tf.keras.Model(inputs=input_m, outputs=output, name="mnist_quantum_model")

        # Model compilation
        model.compile(
          loss='categorical_crossentropy',
          optimizer=tf.keras.optimizers.Adam(learning_rate=0.01) ,
          metrics=[tf.keras.metrics.CategoricalAccuracy()]
        )

       # tf.keras.utils.plot_model(model, to_file='DQNN.png', show_shapes=True, show_layer_names=True)

        # model.summary()
        # Train the model
        model.fit(QX_train, QY_train, epochs=10, batch_size=8, shuffle=True)


        results = model.evaluate(QX_test, QY_test, batch_size=16)

    feature_dim = 2 ** 2  # dimension

    x_train = x_train[:, 0:feature_dim]
    x_test = x_test[:, 0:feature_dim]
    weights, n = np.array([1, 2, 3, 4, 5]), 1

    encoder = qg.BinaryPhaseEncoding(ancilla=True)
    model = qg.BinaryPerceptron(weights=weights)

    measurement = qg.ProbabilityThreshold(qubits=2,
                                          p_zero=False,
                                          threshold=0.3,
                                          labels=[0, 1, 2, 3, 4],
                                          observable=qg.Observable.X())

    full_circuit = qg.combine(x_train[0], encoder, model, measurement)

    
    predict = qg.run(x_test, encoder, model, y_test, measurement)




    tp, tn, fn, fp = 0, 0, 0, 0
    target=y_test
    uni = np.unique(target)
    for j in range(len(uni)):
        c = uni[j]
        for i in range(len(target)):
            if target[i] == c and predict[i] == c:
                tp += 1
            if target[i] != c and predict[i] != c:
                tn += 1
            if (target[i] == c and predict[i] != c):
                fn += 1
            if (target[i] != c and predict[i] == c):
                fp += 1




    acc = (tp + tn) / (tp + fn + tn + fp)
    fpr=  fp/(fp+tn)
    tpr = tp/(tp+fn)

    Acc.append(acc)
    FPR.append(fpr)
    TPR.append(tpr)








