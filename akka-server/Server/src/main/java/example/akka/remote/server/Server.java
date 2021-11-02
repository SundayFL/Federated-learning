package example.akka.remote.server;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import com.typesafe.config.ConfigFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;

public class Server {

    public static void main(String... args) {
        // Creating environment
        ActorSystem system = ActorSystem.create("AkkaRemoteServer", ConfigFactory.load());

        // Create an actor
        ActorRef coordinator = system.actorOf(Props.create(Coordinator.class), "Coordinator");

        // Create an actor
        ActorRef injector = system.actorOf(Props.create(Injector.class), "Injector");
    }
}
