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
    private double DP_noiseVariance;
    private double DP_threshold;

    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof Messages.ClientDataSpread){
            log.info("Reading R values");
            Messages.ClientDataSpread castedMessage = ((Messages.ClientDataSpread) message);
            this.clientId = castedMessage.clientId;
            this.minimum = castedMessage.minimum;
            String encloser = System.getProperty("os.name").startsWith("Windows")?"\"\"":"\"";
            this.publicKeys = castedMessage.contactMap
                    .entrySet()
                    .stream()
                    .collect(Collectors.toMap(participant -> encloser+participant.getKey()+encloser, participant -> participant.getValue().publicKey));

            this.DP_noiseVariance = castedMessage.DP_noiseVariance;
            this.DP_threshold = castedMessage.DP_threshold;
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
                            "--diff_priv", configuration.diffPriv?"True":"False",
                            "--dp_noise_variance", String.valueOf(DP_noiseVariance),
                            "--dp_threshold", String.valueOf(DP_threshold));

            Process process = processBuilder.start();
            int exitCode = process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
