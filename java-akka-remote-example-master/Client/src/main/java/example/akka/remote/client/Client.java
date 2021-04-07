package example.akka.remote.client;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import com.typesafe.config.ConfigFactory;
import example.akka.remote.shared.Messages;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Map;

public class Client {
    public static void main(String[] args) {
        // Creating environment
        ActorSystem system = ActorSystem.create("AkkaRemoteClient", ConfigFactory.load());

        // Client actor
        ActorRef client = system.actorOf(Props.create(ClientActor.class));

        // Send a Calc job
        client.tell(new Messages.StartLearning("mnist"), ActorRef.noSender());
    }
}
