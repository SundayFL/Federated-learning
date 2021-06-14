The Webinar explaining the architecture, implementation and execution of the Sunday-FL platform is available on YouTube at https://www.youtube.com/watch?v=W2sg7cpbxTw

Please cite us if you use our platform. The publication can be found at https://assist-iot.eu/wp-content/uploads/2021/05/ASSIST-IoT-Technical-Report-2-Sunday-FL%E2%80%93Developing-Open-Source-Platform-for-Federated-Learning.pdf

General overview:
Akka enables federated learning here.
Akka projects ==(calls)== &gt Python scripts.

Kindly install the below dependencies before executing python scripts:-
 pip install torch==1.4.0
 pip install torchvision==0.5.0
 pip install syft==0.2.9

These are parameters to execute python scripts. Do substitute the datapath directory with your own directory:-
python server.py --datapath /home/piotr/Desktop/data --id alice --host localhost --port 8777
python client.py --datapath '/home/piotr/Desktop/data' --participantsjsonlist '{"id": "alice", "port": "8777"}' --epochs 10 --modelpath ./saved_model


Location:-
python script server.py located in -&gt fl_image_clasification / pysyft / my / server
python script client.py located in -&gt fl_image_clasification / pysyft / my / client

Kindly ensure JDK, Maven etc are installed before executing the Akka Project:-
These are the parameters to execute akka app. Python commands are hardcoded in the scripts, they need to be altered with respective datapath before execution: -
mvn exec:java -Dexec.mainClass="example.akka.remote.server.Server"
mvn exec:java -Dexec.mainClass="example.akka.remote.client.Client"

Location:-
Client = \java-akka-remote-example-master\Client\src\main\java\example\akka\remote\client
Server = \java-akka-remote-example-master\Server\src\main\java\example\akka\remote\server

 
#do remember to run mvn clean install to resolve any remaining dependency with akka before running the akka project.
The project is working flawlessly in Linux and due to a vendor dependency with TensorFlow the execution of the project on Windows is currently under development.