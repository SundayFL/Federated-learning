FROM python:3.7

#creating directory helloworld in container (linux machine)

RUN mkdir c:\home\client

#copying helloworld.py from local directory to container's helloworld folder

RUN pip install torch
RUN pip install torchvision
RUN pip install syft==0.2.9
RUN pip uninstall websocket
RUN pip install pip install --upgrade websocket_client

COPY client.py /home/client/client.py

#running helloworld.py in container

CMD python /home/client/client.py