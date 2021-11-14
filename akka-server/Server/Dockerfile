# Base Alpine Linux based image with OpenJDK JRE only
FROM openjdk:8-oraclelinux7

ARG JAR_FILE=target/*.jar
COPY ${JAR_FILE} app.jar

COPY target/lib lib
COPY /src/main/python/client go/

RUN chmod 777 /go

ADD src/main/python/data src/main/python/data

EXPOSE 5000

ENTRYPOINT ["java","-cp","app.jar", "example.akka.remote.server.Server"]
