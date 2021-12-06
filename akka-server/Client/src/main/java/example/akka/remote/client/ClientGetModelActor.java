package example.akka.remote.client;

import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.Messages;

import java.io.File;
import java.io.IOException;
import java.util.Map;
import java.util.stream.Collectors;

public class ClientGetModelActor extends UntypedActor {
    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    private String clientId;
    private int minimum;
    private Map<String, Float> publicKeys;

    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof Messages.ClientDataSpread){
            log.info("Reading R values");
            Messages.ClientDataSpread castedMessage = ((Messages.ClientDataSpread) message);
            this.clientId = castedMessage.clientId;
            this.minimum = castedMessage.minimum;
            this.publicKeys = castedMessage.contactMap
                    .entrySet()
                    .stream()
                    .collect(Collectors.toMap(participant -> '"'+participant.getKey()+'"', participant -> participant.getValue().publicKey));

            this.readRValues();
            log.info("R values ready");
            getSender().tell(new Messages.RValuesReady(), getSelf());
        }
    }
    private void readRValues(){
        Configuration.ConfigurationDTO configuration;
        try {
            Configuration configurationHandler = new Configuration();
            configuration = configurationHandler.get();

            // another script
            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.directory(new File(System.getProperty("user.dir")));
            processBuilder
                    .inheritIO()
                    .command("python", configuration.pathToClientLog,
                            "--datapath", configuration.testdatapath,
                            "--id", this.clientId,
                            "--port", String.valueOf(configuration.port),
                            "--public_keys", this.publicKeys.toString(),
                            "--minimum", String.valueOf(this.minimum),
                            "--pathToResources", configuration.pathToResources,
                            "--model_config", configuration.modelConfig,
                            "--epochs", String.valueOf(configuration.epochs),
                            "--save_model", configuration.saveModel?"True":"False");

            Process process = processBuilder.start();
            int exitCode = process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
