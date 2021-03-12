import collections
import numpy as np
import tensorflow_federated as tff
from matplotlib import pyplot as plt
import tensorflow as tf
import pickle
from scipy.ndimage.interpolation import shift

np.random.seed(0)
results = []
print("Num GPUs Available: ", len( tf.config.experimental.list_physical_devices('GPU')))

NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

NUM_CLIENTS = 100
ROUNDS = 10

emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()

example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])

def shift_image(image, dx, dy):
    vv = image.numpy()
    image = image.numpy().reshape((28, 28))
    shifted_image = shift(np.array(image), [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])

def get_class_probability(mask):
    return np.array(mask)/np.sum(mask)


def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def get_mediators(masks, mediator_size, probability_of_all):
    masks_cpy = {i: m for i, m in enumerate(masks)}

    mediators = []
    mediators_masks = []
    mediators_masks_sum = []
    while True:
        mediator = []
        mediator_masks = []
        mediator_mask = np.array([0]*10)  # Sum of masks
        while True:
            if not mediator_mask.any():
                key = list(masks_cpy)[0]
                mask = masks_cpy[key]
                mediator_mask += mask
                mediator.append(key)
                mediator_masks.append(mask)
                masks_cpy.pop(key, None)
            k_min = 1000
            mask_to_add_key = -1
            for key in list(masks_cpy.keys()):  # Find best client
                mask = masks_cpy[key]
                k = kl_divergence(get_class_probability(mediator_mask + mask), probability_of_all)
                if k < k_min:
                    k_min = k
                    mask_to_add_key = key

            mask = masks_cpy[mask_to_add_key]
            mediator_mask += mask
            mediator.append(mask_to_add_key)
            mediator_masks.append(mask)
            masks_cpy.pop(mask_to_add_key, None)

            if len(mediator) == mediator_size:
                break

        mediators.append(mediator)
        mediators_masks.append(mediator_masks)
        mediators_masks_sum.append(mediator_mask)
        if len(masks_cpy) == 0:
            break
    return mediators, mediators_masks_sum

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
        preprocess(client_data.create_tf_dataset_for_client(x))
        for x in client_ids
    ]


sample_clients = emnist_train.client_ids[0: NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

data_masks_tmp = []
sum = np.array([0]*10)

for j in range(NUM_CLIENTS):
    client_dataset = federated_train_data[j]
    plot_data = np.array([0]*10)
    for batch in client_dataset:
        for example in batch['y']:
            # Append counts individually per label to make plots
            # more colorful instead of one color per plot.
            label = example[0].numpy()
            plot_data[label] += 1
        for example in batch['x']:
            shift_image(example, 1, 0)
    sum += plot_data
    data_masks_tmp.append(plot_data)

data_masks, mediators_masks_sum = get_mediators(data_masks_tmp, 10, get_class_probability(sum))
# data_masks = get_simple_mediators(data_masks_tmp, 5)

f = plt.figure(figsize=(12, 7))
for i, mediator_masks_sum in enumerate(mediators_masks_sum):

    plot_data = collections.defaultdict(list)
    for j in range(10):
        for k in range(mediator_masks_sum[j]):
            # Append counts individually per label to make plots
            # more colorful instead of one color per plot.
            label = example[0].numpy()
            plot_data[j].append(j)

    plt.subplot(2, 5, i + 1)
    plt.title('Client {}'.format(i + 1))

    for j in range(10):
        plt.hist(
            plot_data[j],
            density=False,
            bins=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

plt.show()

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

for rep in range(1):


    # print(f"example_dataset: {iter(example_dataset)}")
    # print(f"Length for client: {len(example_dataset)}")

    iterative_process = tff.learning.build_federated_averaging_process(
        model_fn,
        client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
        server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))

    state = iterative_process.initialize()

    for i in range(ROUNDS):

        # print(f"len: {len(federated_train_data)}")
        #
        # counter = 0
        # for i in range(NUM_CLIENTS):
        #     counter += len(federated_train_data[i])
        #     print(len(federated_train_data[i]))
        #
        # print(f"count: {counter}")



        client_ids = data_masks[i]
        sample_clients = [emnist_train.client_ids[cid] for cid in client_ids]
        federated_train_data = make_federated_data(emnist_train, sample_clients)


        state, metrics = iterative_process.next(state, federated_train_data)
        print(f'rep: {rep+1} round  {i+1}')

        # Test
        evaluation = tff.learning.build_federated_evaluation(model_fn)
        federated_test_data = make_federated_data(emnist_test, sample_clients)
        test_metrics = evaluation(state.model, federated_test_data)
        print(f"Test {NUM_CLIENTS} {ROUNDS} {rep}: {str(test_metrics['sparse_categorical_accuracy'])}")
        results.append(test_metrics['sparse_categorical_accuracy'])


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

    print(f"Test {NUM_CLIENTS} {ROUNDS} {rep}: {str(test_metrics['sparse_categorical_accuracy'])}")
    results.append(test_metrics['sparse_categorical_accuracy'])

print(results)

for result in results:
    print(result)