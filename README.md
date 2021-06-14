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
 
pip install syft==0.2.9
```

## Usage
Location:-

python script server.py located in -> fl_image_clasification / pysyft / my / server

python script client.py located in -> fl_image_clasification / pysyft / my / client
```python
python server.py --datapath /home/piotr/Desktop/data --id alice --host localhost --port 8777
python client.py --datapath '/home/piotr/Desktop/data' --participantsjsonlist '{"id": "alice", "port": "8777"}' --epochs 10 --modelpath ./saved_model
```

Location:-

Client = \java-akka-remote-example-master\Client\src\main\java\example\akka\remote\client

Server = \java-akka-remote-example-master\Server\src\main\java\example\akka\remote\server

Kindly ensure JDK, Maven etc are installed before executing the Akka Project:-
```java
mvn exec:java -Dexec.mainClass="example.akka.remote.server.Server"
mvn exec:java -Dexec.mainClass="example.akka.remote.client.Client"
```
Do remember to run mvn clean install to resolve any remaining dependency with akka before running the akka project.
## Project status
The project is working flawlessly in Linux. Due to a vendor dependency with TensorFlow the execution of the project on Windows is currently under development.

## License
[MIT](https://choosealicense.com/licenses/mit/)
