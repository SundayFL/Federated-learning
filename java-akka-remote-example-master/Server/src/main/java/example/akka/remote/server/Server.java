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

        /*System.out.println("Working Directory = " + System.getProperty("user.dir"));

        ProcessBuilder processBuilder2 = new ProcessBuilder();
        processBuilder2.directory(new File(System.getProperty("user.dir")));
        System.out.println("Before ls");
        processBuilder2.inheritIO().command("ls", "-l", "go");
        try {
            Process process2 = processBuilder2.start();
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("After ls");
        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.directory(new File(System.getProperty("user.dir")));

        processBuilder.inheritIO().command("./go/client", "--datapath", "/home/piotr/Desktop/data"); // ./src/main/python/client/client
        try {
            System.out.println("Before start");
            Process process = processBuilder.start();
            System.out.println("After start");
            int exitCode = process.waitFor();
            System.out.println("After execution");
            BufferedReader read = new BufferedReader(new InputStreamReader(
                    process.getInputStream()));
            while (read.ready()) {
                System.out.println(read.readLine());
            }

            System.out.println("Error:");

            BufferedReader readError = new BufferedReader(new InputStreamReader(
                    process.getErrorStream()));
            while (readError.ready()) {
                System.out.println(readError.readLine());
            }

        } catch (IOException | InterruptedException e) {
            System.out.println("Message: " + e.getMessage());
            e.printStackTrace();
        }*/
        // Creating environment
        ActorSystem system = ActorSystem.create("AkkaRemoteServer", ConfigFactory.load());

        // Create an actor
        ActorRef coordinator = system.actorOf(Props.create(Coordinator.class), "Coordinator");
    }
}
