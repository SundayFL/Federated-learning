# Base Alpine Linux based image with OpenJDK JRE only
FROM openjdk:8-oraclelinux7

ARG JAR_FILE=target/*.jar
COPY ${JAR_FILE} app.jar

COPY target/lib lib
COPY /src/main/python/client.py go/

RUN chmod 777 /go

ADD src/main/python src/main/python
ADD src/main/resources src/main/resources

EXPOSE 5000

ENTRYPOINT ["java","-cp","app.jar", "example.akka.remote.server.Server", "1"]
# add "1" to list to enable secure aggregation
