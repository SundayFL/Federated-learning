package example.akka.remote.client;

import akka.actor.ActorSelection;
import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.Messages;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.LocalDateTime;

public class ClientRunModuleActor extends UntypedActor {
    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof ClientActor.RunModule) {
            log.info("Received RunModule command");
            this.runLearning2();
        }
    }

    private void runLearning2() {
        System.out.println("Working Directory = " + System.getProperty("user.dir"));

        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.directory(new File(System.getProperty("user.dir")));
        //processBuilder.command("ls");
        //processBuilder.command("./src/main/python/server/server");
        processBuilder.inheritIO().command("python", "./src/main/python/server.py", "--datapath",
                "./src/main/python/data", "--id", "alice", "--host", "localhost", "--port", "8777");
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
        }
    }
}
