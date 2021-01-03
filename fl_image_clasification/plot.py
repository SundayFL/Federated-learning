import matplotlib.pyplot as plt

x = [10, 20, 30, 50, 100]
xi = list(range(len(x)))

## 1 ##

# client1 = [9.09,	14.55,	24.55,	61.82,	68.39]
# client4 = [26.22,	46.89,	58.87,	68.67,	85.78]
client6 = [38.99,	57.25,	67.97,	76.96,	81.88]
client10 = [39.74,	62.93,	69.74,	78.28,	82.59]
client20 = [52.12,	71.08,	76.23,	81.08,	85.58]
#
#
# plt.plot(xi, client1, label="1 client")
# plt.plot(xi, client4, label="4 clients")
# plt.plot(xi, client6, label="6 clients")
# plt.plot(xi, client10, label="10 clients")
# plt.plot(xi, client20, label="20 clients")
#
# plt.xticks(xi, x)
# plt.ylabel('Accuracy')
# plt.legend(loc="upper left")
# plt.title('Accuracy depending on number of clients')
# plt.show()

## 2 ##
# client_2_3 = [50.43,	68.55,	76.96,	81.45,	85.65]
# client_2_5 = [53.02,	73.62,	77.41,	80.78,	85.78]
# client_2_10 = [68.23,	78.27,	83.42,	86.41,	89.13]
#
# plt.plot(xi, client20, label="20 clients")
# plt.plot(xi, client_2_10, label="10 clents")
#
# plt.xticks(xi, x)
# plt.legend(loc="upper left")
#
# plt.ylabel('Accuracy')
# plt.title('Accuracy depending on data distribution')
# plt.show()

## 3 ##
# client_3 = [61.41,	75.86,	80.66,	83.70,	87.36]
#
# plt.plot(xi, client10, label="Normal data")
# plt.plot(xi, client_3, label="Unbalanced data")
#
# plt.xticks(xi, x)
# plt.legend(loc="upper left")
# plt.ylabel('Accuracy')
# plt.title('Accuracy depending on data distribution')
# plt.show()

## 4 ##
# client_4 = [61.41,	75.86,	80.66,	83.70,	87.36]
#
# plt.plot(xi, client10, label="Normal data")
# plt.plot(xi, client_4, label="Unbalanced data")
#
# plt.xticks(xi, x)
# plt.legend(loc="upper left")
# plt.ylabel('Accuracy')
# plt.title('Accuracy depending on data distribution')
# plt.show()

## 5 ##
# client_1 = [41.24,	60.17,	66.21,	70.22,	75.38]
# client_2 = [38.00,	50.71,	58.53,	63.03,	67.60]
# client_3 = [31.36,	49.17,	53.55,	58.19,	62.05]
#
# plt.plot(xi, client10, label="Normal clients")
# plt.plot(xi, client_1, label="1 fake client")
# plt.plot(xi, client_2, label="2 fake client")
# plt.plot(xi, client_3, label="3 fake client")
#
# plt.xticks(xi, x)
# plt.legend(loc="upper left")
# plt.ylabel('Accuracy')
# plt.title('Accuracy depending on number of fake clients')
# plt.show()

## 6 ##
client_6 = [10.34,	10.86,	9.22,	8.10,	10.26]

plt.plot(xi, client10, label="Normal training")
plt.plot(xi, client_6, label="Training without model aggregating")

plt.xticks(xi, x)
plt.legend(loc="upper left")
plt.ylabel('Accuracy')
plt.xlabel('Number of training rounds')
plt.title('Accuracy depending on model aggregation')
plt.show()