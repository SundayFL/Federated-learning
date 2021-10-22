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
        // Message that says to run the module
        if (message instanceof ClientActor.RunModule) {
            log.info("Received RunModule command");
            ClientActor.RunModule receivedMessage = (ClientActor.RunModule) message;
            this.runLearning(receivedMessage.moduleFileName, receivedMessage.modelConfig);
        }
    }

    // Runs module
    private void runLearning(String moduleFileName, String modelConfig) {
        Configuration.ConfigurationDTO configuration;
        try {
            Configuration configurationHandler = new Configuration();
            configuration = configurationHandler.get();

            // execute scrips with proper parameters
            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.directory(new File(System.getProperty("user.dir")));
            log.info( configuration.pathToModules + moduleFileName);
            processBuilder
                .inheritIO()
                .command("python", configuration.pathToModules + moduleFileName,
                         "--datapath", configuration.datapath,
                         "--data_file_name", configuration.datafilename,
                         "--target_file_name", configuration.targetfilename,
                         "--id", configuration.id,
                         "--host", configuration.host,
                         "--port", String.valueOf(configuration.port),
                         "--data_set_id", String.valueOf(configuration.dataSetId),
                         "--model_config", modelConfig);

            Process process = processBuilder.start();
            int exitCode = process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
