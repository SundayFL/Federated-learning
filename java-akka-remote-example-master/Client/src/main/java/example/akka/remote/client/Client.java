package example.akka.remote.client;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import com.typesafe.config.ConfigFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Map;

public class Client {
    public static void main(String[] args) {
        /*System.out.println("Working Directory = " + System.getProperty("user.dir"));

        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.directory(new File(System.getProperty("user.dir")));
        //processBuilder.command("ls");
        //processBuilder.command("./src/main/python/server/server");
        processBuilder.inheritIO().command("./src/main/python/server/server", "--datapath", "/home/piotr/Desktop/data");
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
            e.printStackTrace();
        }*/

        /*try {
            System.out.println("Before run");
            Process proc = Runtime.getRuntime().exec("./src/main/python/server/server --datapath '/home/piotr/Desktop/data'");
            System.out.println("After exec");
            try {
                proc.waitFor();
                System.out.println("After wait");
            } catch (InterruptedException e) {
                System.out.println(e.getMessage());
            }

            BufferedReader read = new BufferedReader(new InputStreamReader(
                    proc.getInputStream()));
            while (read.ready()) {
                System.out.println(read.readLine());
            }

            System.out.println("Error:");

            BufferedReader readError = new BufferedReader(new InputStreamReader(
                    proc.getErrorStream()));
            while (readError.ready()) {
                System.out.println(readError.readLine());
            }
        } catch (IOException e) {
            System.out.println(e.getMessage());
        }*/


        // Creating environment
        ActorSystem system = ActorSystem.create("AkkaRemoteClient", ConfigFactory.load());

        // Client actor
        ActorRef client = system.actorOf(Props.create(ClientActor.class));

        // Send a Calc job
        client.tell("Start", ActorRef.noSender());
    }
}
