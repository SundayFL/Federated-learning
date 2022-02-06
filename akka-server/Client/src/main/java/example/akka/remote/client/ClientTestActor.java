package example.akka.remote.client;

import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.Messages;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;
import java.util.stream.Collectors;

public class ClientTestActor extends UntypedActor {
    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    private byte[] model;
    private byte[] results;

    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof Messages.TestMyModel){
            Configuration.ConfigurationDTO configuration;
            Configuration configurationHandler = new Configuration();
            configuration = configurationHandler.get();
            log.info("Model test");
            Messages.TestMyModel castedMessage = ((Messages.TestMyModel) message);
            // prepare arguments to pass to the script
            this.model = castedMessage.bytes;
            this.testModel();
            log.info("Model tested");
            results = Files.readAllBytes(Paths.get(configuration.pathToResources+configuration.id+".txt"));
            File tempfile = new File(configuration.pathToResources+configuration.id+".txt");
            boolean deleted = tempfile.delete();
            getSender().tell(new Messages.TestResults(this.results, configuration.id), getSelf());
        }
    }
    private void testModel(){
        Configuration.ConfigurationDTO configuration;
        Configuration configurationHandler = new Configuration();
        try {
            configuration = configurationHandler.get();
            Files.write(Paths.get(configuration.pathToResources+configuration.id+"_saved_model"), this.model);

            // execute script in order to retrieve weights and make R values of them
            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.directory(new File(System.getProperty("user.dir")));
            processBuilder
                    .inheritIO()
                    .command("python", configuration.pathToTesting,
                            "--datapath", configuration.testdatapath,
                            "--id", configuration.id,
                            "--port", String.valueOf(configuration.port),
                            "--pathToResources", configuration.pathToResources,
                            "--model_config", configuration.modelConfig,
                            "--modelpath", configuration.pathToResources+configuration.id+"_saved_model");

            Process process = processBuilder.start();
            int exitCode = process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
