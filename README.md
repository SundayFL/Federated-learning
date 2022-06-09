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


# How to create a docker image of the server

1. Sign in to Docker:

```
sudo docker login
```

2. Enter the akka-server/Server directory and build the image:

```
sudo docker build -t server .
```

The process of building the docker image uses a prepared Dockerfile in the directory. The fifth (optional parameter) in the last line (_ENTRYPOINT_ command) serves to enable/disable security modules and is required to be entered as a string containing two numbers (e.g. "1 0.5"). You may omit the parameter if you intend to disable both modules. Otherwise:
* enter 1 as the first number to enable secure aggregation, 0 to disable;
* enter the threshold value to be used in the differential privacy module as the second number; exclude the number if you're not setting it.

Examples:

```
ENTRYPOINT ["java","-cp","app.jar", "example.akka.remote.server.Server"]
# neither module used
ENTRYPOINT ["java","-cp","app.jar", "example.akka.remote.server.Server", "1"]
# secure aggregation used
ENTRYPOINT ["java","-cp","app.jar", "example.akka.remote.server.Server", "1 0.5"]
# both modules used, differential privacy threshold set to 0.5
```

3. To run the server from docker image, enter:

```
sudo docker run server:latest
```

# How to run FL server remotely using Azure

## Create a docker container on Azure

1. Make a resource group on Azure, e.g. **flresources**.
2. Make a container registry on Azure, e.g. **flregistry**. Make sure its name is unique.
3. Enter the container registry on Microsoft Azure. Enter Settings/Access keys section and enable admin user. Make sure to save the generated password that appears below.
4. Run mvn clean install in akka-server directory and its subdirectories Server and Client. Build a docker image in the akka-server/Server directory:

```
az acr build -t fl/server -r flregistry .
```

where -r denotes the container registry name. You might be required to enter your Azure credentials.
5. Specify a DNS name to enable connection to the server, e.g. **sundayfltest**. In the file akka-server/Server/src/main/resources/application.conf, change the parameter _hostname_ in akka --> remote --> netty.tcp field from '127.0.0.1' to '**sundayfltest**.eastus.azurecontainer.io'.
6. Create a container instance through terminal:

```
az container create -n [container instance name] -g [resource group name] --image [container registry name].azurecr.io/fl/server:latest --ports [list of ports separated by space; 5000 required] --dns-name-label [DNS name] --registry-username [admin user name; default is the same as the container registry name] --registry-password [saved admin password to the registry] --cpu 2 --memory 3
```

Example with parameters:

```
az container create -n flcontainer -g flresources --image flregistry.azurecr.io/fl/server:latest --ports 80 2552 2553 2554 2555 5000 --dns-name-label sundayfltest --registry-username flregistry --registry-password [saved password] --cpu 2 --memory 3
```

You can specify different values of the cpu and memory parameters; 2 and 3 are the most recommended once. Remember to list 5000 in ports.
7. Enter the Azure page of the created container instance. In Settings/Containers section, enter the Connect part and connect to the container. Sunday-FL requires Python 3.8 to work, which is not installed in Azure container instances. Enter the following commands to install Python 3.8:

```
yum install -y gcc openssl-devel bzip2-devel libffi-devel zlib-devel wget make
cd /opt
wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
tar xzf Python-3.8.12.tgz
cd Python-3.8.12
./configure --enable-optimizations
make altinstall
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.8 get-pip.py
pip install torch==1.4.0 --no-cache-dir
pip install torchvision==0.5.0 syft==0.2.5 argparse pathlib asyncio
pip install logging
```

Installation regularly takes from 5 to 10 minutes. This step is required every time the container instance is started; putting the commands in Dockerfile with RUN might not work.

## Connecting clients to remote server

1. In the file akka-server/Client/src/main/resources/appConfig.json, change the _address_ parameter from '127.0.0.1:5000' to '**sundayfltest**.eastus.azurecontainer.io:5000', which consists of the DNS address and a port.
2. Enter the command in the akka-server/Client directory (having run mvn clean install!):

```
mvn exec:java -Dexec.mainClass="example.akka.remote.client.Client" -Dexec.args="2552 0 alice 8778 0.5"
```

The -Dexec.args parameters are:
* AKKA port (make sure it is not in use yet);
* the dataset partition index;
* your client id;
* Websocket port (make sure it is not in use yet);
* variance to be used when enabling the differential privacy module; enter 0 to disable the module.
3. To exit the client, press Ctrl-C (preferably after the learning round ends).

## PS

In order to run FL locally, change the mentioned configuration parameters back to 127.0.0.1 and run mvn clean install.




## Project status
The project is working flawlessly in Linux and Windows.
The Verificator module is currently under development.

## License
[MIT](https://choosealicense.com/licenses/mit/)
