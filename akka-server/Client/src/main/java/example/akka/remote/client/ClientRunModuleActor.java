package example.akka.remote.client;

import akka.actor.ActorPath;
import akka.actor.ActorRef;
import akka.actor.ActorSelection;
import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import example.akka.remote.shared.Messages;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.util.*;
import java.util.function.Consumer;
import java.util.stream.Collectors;

public class ClientRunModuleActor extends UntypedActor {
    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    private String clientId;
    private int minimum;
    private Map<String, String> addresses; // ???? P
    private Map<String, Integer> ports; // ???? P
    private List<Float> publicKeys; // ???? P

    private Map<String, Messages.ParticipantData> roundParticipants;



    @Override
    public void onReceive(Object message) throws Exception {
        // Message that says to run the module
        if (message instanceof Messages.RunModule) {
            log.info("Received RunModule command");
            Messages.RunModule receivedMessage = (Messages.RunModule) message;
            this.runLearning(receivedMessage.moduleFileName, receivedMessage.modelConfig);
        }


        // P, already done via  'roundParticipants' field

        if (message instanceof Messages.ClientDataSpread){
            this.clientId = ((Messages.ClientDataSpread) message).clientId;
            this.minimum = ((Messages.ClientDataSpread) message).minimum;
            this.addresses = ((Messages.ClientDataSpread) message).addresses;
            this.ports = ((Messages.ClientDataSpread) message).ports;
            this.publicKeys = ((Messages.ClientDataSpread) message).publicKeys;

            this.readRValues();
            getSender().tell(new Messages.RValuesReady(), getSelf());
        }



    }

    // Runs module
    private void runLearning(String moduleFileName, String modelConfig) {
        Configuration.ConfigurationDTO configuration;
        try {
            Configuration configurationHandler = new Configuration();
            configuration = configurationHandler.get();

            String participantsJson = getParticipantsJson();
            String tempvar = participantsJson.replace('"', '\'');

            // execute scripts with proper parameters
            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.directory(new File(System.getProperty("user.dir")));
            log.info(configuration.pathToModules + moduleFileName);
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
                            "--model_config", modelConfig,
                            "--participantsjsonlist", tempvar);

            Process process = processBuilder.start();
            int exitCode = process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    //P, IN PYTHON !!!

    private void readRValues(){
        Configuration.ConfigurationDTO configuration;
        try {
            Configuration configurationHandler = new Configuration();
            configuration = configurationHandler.get();

            List<String> ids = new ArrayList<>(this.addresses.keySet());
            Collections.sort(ids);

            // another script
            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.directory(new File(System.getProperty("user.dir")));
            processBuilder
                    .inheritIO()
                    .command("python", configuration.pathToClientLog,
                            "--datapath", configuration.testdatapath,
                            "--id", this.clientId,
                            "--port", String.valueOf(configuration.port),
                            "--foreign_ids", "["+ids.stream().map(Object::toString).collect(Collectors.joining(", "))+"]",
                            "--public_keys", "["+this.publicKeys.stream().map(Object::toString).collect(Collectors.joining(", "))+"]",
                            "--minimum", String.valueOf(this.minimum),
                            "--pathToResources", configuration.pathToResources,
                            "--epochs", String.valueOf(configuration.epochs));

            Process process = processBuilder.start();
            int exitCode = process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }




    // Returns participates data as a json
    private String getParticipantsJson() {
        ObjectMapper mapper = new ObjectMapper();
        try {

            List<LearningData> listToSerialize = new ArrayList<>();
            this.roundParticipants.entrySet().stream()
                    .forEach(pd -> listToSerialize.add(new LearningData(pd.getKey(), pd.getValue().port)));

            String json = mapper.writeValueAsString(listToSerialize);
            System.out.println("json -> " + json);
            return json;
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }
        return "";
    }

    // Class for serializing modules list
    public class LearningData {
        public LearningData(String id, int port) {
            this.id = id;
            this.port = port;
        }

        public String id;
        public int port;
    }

}
