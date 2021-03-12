import collections
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import pickle

np.random.seed(0)
results = []
data_masks_tmp = []

for rep in range(1):
    emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

    example_dataset = emnist_train.create_tf_dataset_for_client(
        emnist_train.client_ids[0])

    # print(f"example_dataset: {iter(example_dataset)}")
    # print(f"Length for client: {len(example_dataset)}")

    NUM_EPOCHS = 5
    BATCH_SIZE = 20
    SHUFFLE_BUFFER = 100
    PREFETCH_BUFFER = 10

    NUM_CLIENTS = 10
    ROUNDS = 0

    def make_federated_data_prepared(client_data, client_ids, number):
        def filter_fn(elem):
            return tf.math.equal(elem["label"], number)

        additional_dataset = client_data.create_tf_dataset_for_client(client_ids[0])
        for i in range(1, len(client_ids)):
            additional_dataset = additional_dataset.concatenate(client_data.create_tf_dataset_for_client(client_ids[i]))

        additional_dataset = additional_dataset.filter(filter_fn)

        return additional_dataset

    sample_clients_additional = emnist_train.client_ids[NUM_CLIENTS+1:NUM_CLIENTS+11]
    federated_train_data_additional = make_federated_data_prepared(emnist_train, sample_clients_additional, 0)



    def preprocess(dataset):

      def batch_format_fn(element):
        """Flatten a batch `pixels` and return the features as an `OrderedDict`."""
        return collections.OrderedDict(
            x=tf.reshape(element['pixels'], [-1, 784]),
            y=tf.reshape(element['label'], [-1, 1]))

      return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
          BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)


    preprocessed_example_dataset = preprocess(example_dataset)


    def make_federated_data(client_data, client_ids):
      return [
          preprocess(client_data.create_tf_dataset_for_client(x).concatenate(federated_train_data_additional)) if x == client_ids[0]
          else preprocess(client_data.create_tf_dataset_for_client(x))
          for x in client_ids
      ]


    sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
    federated_train_data = make_federated_data(emnist_train, sample_clients)

    # f = plt.figure(figsize=(12, 7))
    # f.suptitle('Label Counts for a Sample of Clients')
    # client_dataset = federated_train_data_additional
    # plot_data = collections.defaultdict(list)
    # for batch in client_dataset:
    #     for example in batch['y']:
    #       # Append counts individually per label to make plots
    #       # more colorful instead of one color per plot.
    #       label = example[0].numpy()
    #       plot_data[label].append(label)
    # plt.subplot(2, 3, 1)
    # plt.title('Client {}'.format(0))
    # for j in range(10):
    #     plt.hist(
    #         plot_data[j],
    #         density=False,
    #         bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    #
    # plt.show()

    f = plt.figure(figsize=(12, 7))
    for i in range(NUM_CLIENTS):
      client_dataset = federated_train_data[i]
      plot_data = collections.defaultdict(list)
      for batch in client_dataset:
        for example in batch['y']:
          # Append counts individually per label to make plots
          # more colorful instead of one color per plot.
          label = example[0].numpy()
          plot_data[label].append(label)
      plt.subplot(2, 5, i+1)
      plt.title('Client {}'.format(i+1))
      for j in range(10):
        plt.hist(
            plot_data[j],
            density=False,
            bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    plt.show()

    # print(f"len: {len(federated_train_data)}")
    #
    # counter = 0
    # for i in range(NUM_CLIENTS):
    #     counter += len(federated_train_data[i])
    #     print(len(federated_train_data[i]))
    #
    # print(f"count: {counter}")


    def create_keras_model():
      return tf.keras.models.Sequential([
          tf.keras.layers.Input(shape=(784,)),
          # tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.Dense(10, activation='softmax')
      ])


    def model_fn():
      # We _must_ create a new model here, and _not_ capture it from an external
      # scope. TFF will call this within different graph contexts.
      keras_model = create_keras_model()
      return tff.learning.from_keras_model(
          keras_model,
          input_spec=preprocessed_example_dataset.element_spec,
          loss=tf.keras.losses.SparseCategoricalCrossentropy(),
          metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])


    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))


    state = iterative_process.initialize()

    for i in range(ROUNDS):
        state, metrics = iterative_process.next(state, federated_train_data)
        print(f'rep: {rep+1} round  {i+1}')


    MnistVariables = collections.namedtuple(
        'MnistVariables', 'weights bias num_examples loss_sum accuracy_sum')

    def create_mnist_variables():
      return MnistVariables(
          weights=tf.Variable(
              lambda: tf.zeros(dtype=tf.float32, shape=(784, 10)),
              name='weights',
              trainable=True),
          bias=tf.Variable(
              lambda: tf.zeros(dtype=tf.float32, shape=(10)),
              name='bias',
              trainable=True),
          num_examples=tf.Variable(0.0, name='num_examples', trainable=False),
          loss_sum=tf.Variable(0.0, name='loss_sum', trainable=False),
          accuracy_sum=tf.Variable(0.0, name='accuracy_sum', trainable=False))

    def mnist_forward_pass(variables, batch):
      y = tf.nn.softmax(tf.matmul(batch['x'], variables.weights) + variables.bias)
      predictions = tf.cast(tf.argmax(y, 1), tf.int32)

      flat_labels = tf.reshape(batch['y'], [-1])
      loss = -tf.reduce_mean(
          tf.reduce_sum(tf.one_hot(flat_labels, 10) * tf.math.log(y), axis=[1]))
      accuracy = tf.reduce_mean(
          tf.cast(tf.equal(predictions, flat_labels), tf.float32))

      num_examples = tf.cast(tf.size(batch['y']), tf.float32)

      variables.num_examples.assign_add(num_examples)
      variables.loss_sum.assign_add(loss * num_examples)
      variables.accuracy_sum.assign_add(accuracy * num_examples)

      return loss, predictions


    def get_local_mnist_metrics(variables):
      return collections.OrderedDict(
          num_examples=variables.num_examples,
          loss=variables.loss_sum / variables.num_examples,
          accuracy=variables.accuracy_sum / variables.num_examples)


    @tff.federated_computation
    def aggregate_mnist_metrics_across_clients(metrics):
      return collections.OrderedDict(
          num_examples=tff.federated_sum(metrics.num_examples),
          loss=tff.federated_mean(metrics.loss, metrics.num_examples),
          accuracy=tff.federated_mean(metrics.accuracy, metrics.num_examples))


    class MnistModel(tff.learning.Model):

      def __init__(self):
        self._variables = create_mnist_variables()

      @property
      def trainable_variables(self):
        return [self._variables.weights, self._variables.bias]

      @property
      def non_trainable_variables(self):
        return []

      @property
      def local_variables(self):
        return [
            self._variables.num_examples, self._variables.loss_sum,
            self._variables.accuracy_sum
        ]

      @property
      def input_spec(self):
        return collections.OrderedDict(
            x=tf.TensorSpec([None, 784], tf.float32),
            y=tf.TensorSpec([None, 1], tf.int32))

      @tf.function
      def forward_pass(self, batch, training=True):
        del training
        loss, predictions = mnist_forward_pass(self._variables, batch)
        num_exmaples = tf.shape(batch['x'])[0]
        return tff.learning.BatchOutput(
            loss=loss, predictions=predictions, num_examples=num_exmaples)

      @tf.function
      def report_local_outputs(self):
        return get_local_mnist_metrics(self._variables)

      @property
      def federated_output_computation(self):
        return aggregate_mnist_metrics_across_clients


    evaluation = tff.learning.build_federated_evaluation(model_fn)

    federated_test_data = make_federated_data(emnist_test, sample_clients)

    test_metrics = evaluation(state.model, federated_test_data)

    print(str(test_metrics))
    results.append(test_metrics['sparse_categorical_accuracy'])

print(results)

for result in results:
    print(result)
