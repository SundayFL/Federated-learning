package example.akka.remote.client;

import akka.actor.ActorSelection;
import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import com.fasterxml.jackson.databind.ObjectMapper;
import example.akka.remote.shared.Messages;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class ClientRunModuleActor extends UntypedActor {
    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    private String clientId;
    private int numberOfClientstoAwait, minimum;
    private Object RValues;
    private Map<String, Object> ownRValues;
    private Map<String, String> addresses;
    private Map<String, Integer> ports;
    private List<Float> publicKeys;

    public <K, V> Map<K, V> getMeOut(Map<K, V> m){
        return m.entrySet().stream().filter(entry -> !entry.getKey().equals(clientId)).collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    @Override
    public void onReceive(Object message) throws Exception {
        // Message that says to run the module
        if (message instanceof ClientActor.RunModule) {
            log.info("Received RunModule command");
            ClientActor.RunModule receivedMessage = (ClientActor.RunModule) message;
            this.runLearning(receivedMessage.moduleFileName, receivedMessage.modelConfig);
        }
        if (message instanceof Messages.ClientDataSpread){
            this.clientId = ((Messages.ClientDataSpread) message).clientId;
            this.numberOfClientstoAwait = ((Messages.ClientDataSpread) message).numberOfClients;
            this.minimum = ((Messages.ClientDataSpread) message).minimum;
            this.addresses = getMeOut(((Messages.ClientDataSpread) message).addresses);
            this.ports = getMeOut(((Messages.ClientDataSpread) message).ports);
            this.publicKeys = ((Messages.ClientDataSpread) message).publicKeys;

             /* do we need it?
            ObjectMapper mapper = new ObjectMapper();
            Map<String, Object> JSONmap = new HashMap<>();
            JSONmap.put("addresses", addresses);
            JSONmap.put("ports", ports);
            JSONmap.put("publicKeys", publicKeys);
            mapper.writeValue(new File("./src/main/resources/"+clientId+".json"), JSONmap);
             */
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
                         "--foreign_addresses", addresses.toString(),
                         "--foreign_ports", ports.toString(),
                         "--public_keys", publicKeys.toString(),
                         "--min_devices", String.valueOf(minimum),
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
