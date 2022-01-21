# Sunday - FL

The Webinar explaining the architecture, implementation and execution of the Sunday-FL platform is available on [YouTube](https://www.youtube.com/watch?v=W2sg7cpbxTw)

Please cite us if you use our platform. The publication can be found [here](https://assist-iot.eu/wp-content/uploads/2021/05/ASSIST-IoT-Technical-Report-2-Sunday-FL%E2%80%93Developing-Open-Source-Platform-for-Federated-Learning.pdf)

## General overview

Akka enables federated learning here.

Akka projects ==(calls)== > Python scripts.

## Installation

Kindly install the below dependencies before executing python scripts:-
 
```bash 
pip install torch==1.4.0
 
pip install torchvision==0.5.0
 
pip install syft==0.2.5
```

## Usage

Location:-

Client = \java-akka-remote-example-master\Client\src\main\java\example\akka\remote\client

Server = \java-akka-remote-example-master\Server\src\main\java\example\akka\remote\server

Kindly ensure JDK, Maven etc are installed before executing the Akka Project:-
```java
mvn exec:java -Dexec.mainClass="example.akka.remote.server.Server"
mvn exec:java -Dexec.mainClass="example.akka.remote.client.Client"
```

Use the below for runing multiple clients:-


e.g.-  2553 - akka port,
       6 id of the dataset due to TFF indexing,
       bob - client id
       8778 python port
       0.5 differential privacy(security module) variance, if 0 - no differential privacy
```java
mvn exec:java -Dexec.mainClass="example.akka.remote.client.Client" -Dexec.args="2553 6 bob 8778 0.5"
```

Use the below for runing server:-

e.g.-  1 - secure aggregation enabled, if 0 - disabled,
       0.2 - differential privacy(security module) threshold  [OPTIONAL ARGUMENT],
```java
mvn exec:java -Dexec.mainClass="example.akka.remote.server.Server" -Dexec.args="1 0.2"
```

Do remember to run mvn clean install to resolve any remaining dependency with akka before running the akka project.
## Project status
The project is working flawlessly in Linux and Windows.
The Verificator module is currently under development.

## License
[MIT](https://choosealicense.com/licenses/mit/)
