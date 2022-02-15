package example.akka.remote.client;

import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.Messages;

import java.io.File;
import java.io.IOException;

public class ClientGetModelActor extends UntypedActor {
    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    private String clientId;
    private double DP_std;
    private double DP_threshold;
    private boolean secureAgg;
    private boolean diffPriv;
    private boolean learningTaskId;

    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof Messages.ClientDataSpread){
            log.info("Reading R values");
            Messages.ClientDataSpread castedMessage = ((Messages.ClientDataSpread) message);
            // prepare arguments to pass to the script
            this.clientId = castedMessage.clientId; // client id
            String encloser = System.getProperty("os.name").startsWith("Windows")?"\"\"":"\"";
            // Windows and Linux handle parsing maps differently when it comes to quotation marks
            // public keys extracted to another map

            // differential privacy arguments
            this.secureAgg = castedMessage.secureAgg;
            this.diffPriv = castedMessage.diffPriv;
            this.DP_threshold = castedMessage.DP_threshold;
            this.DP_std = castedMessage.DP_std;
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

            // execute script in order to retrieve weights and make R values of them
            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.directory(new File(System.getProperty("user.dir")));
            processBuilder
                    .inheritIO()
                    .command("python", configuration.pathToLocalModel,
                            "--datapath", configuration.testdatapath,
                            "--id", this.clientId,
                            "--port", String.valueOf(configuration.port),
                            "--pathToResources", configuration.pathToResources,
                            "--model_config", configuration.modelConfig,
                            "--epochs", String.valueOf(configuration.epochs),
                            "--diff_priv", configuration.diffPriv?"True":"False",
                            "--dp_noise_std", String.valueOf( DP_std ),
                            "--dp_threshold", String.valueOf(DP_threshold),
                            "--learningTaskId", String.valueOf(configuration.learningTaskId));

            Process process = processBuilder.start();
            int exitCode = process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }
}
