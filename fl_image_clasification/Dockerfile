FROM python:3.7

#creating directory helloworld in container (linux machine)

RUN mkdir c:\home\tff_server

#copying helloworld.py from local directory to container's helloworld folder

RUN pip install tensorflow==2.2.1

RUN pip install tensorflow-federated==0.16.1

COPY server.py /home/tff_server/server.py

#running helloworld.py in container

CMD python /home/tff_server/server.py