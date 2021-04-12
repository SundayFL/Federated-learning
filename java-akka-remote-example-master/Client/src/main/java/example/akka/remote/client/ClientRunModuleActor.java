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
            this.runLearning(((ClientActor.RunModule) message).moduleFileName);
        }
    }

    private void runLearning(String moduleFileName) {
        Configuration.ConfigurationDTO configuration;
        try {
            configuration = Configuration.get();

            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.directory(new File(System.getProperty("user.dir")));
            processBuilder
                .inheritIO()
                .command("python", configuration.pathToModules + moduleFileName,
                         "--datapath", configuration.datapath,
                         "--id", configuration.id,
                         "--host", configuration.host,
                         "--port", String.valueOf(configuration.port));

            Process process = processBuilder.start();
            int exitCode = process.waitFor();
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
