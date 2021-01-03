import collections
import time

import tensorflow as tf
import tensorflow_federated as tff

import grpc

executor_factory = tff.framework.local_executor_factory(10)

tff.simulation.run_server(executor_factory, 10, 80)
