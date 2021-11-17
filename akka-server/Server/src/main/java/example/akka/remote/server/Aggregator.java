package example.akka.remote.server;

import akka.actor.*;
import akka.actor.dsl.Creators;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.remote.transport.ThrottlerTransportAdapter;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import example.akka.remote.shared.LoggingActor;
import example.akka.remote.shared.Messages;
import scala.concurrent.duration.FiniteDuration;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

import static example.akka.remote.shared.Messages.*;

public class Aggregator extends UntypedActor {

    public Aggregator(ActorRef coordinator) {
        log.info("Selector created");
        this.coordinator = coordinator;
        tickActor = getContext().system().actorOf(Props.create(Ticker.class), "Ticker");
        log.info("coordinator -> " + coordinator.path());
    }

    // Participants taking part in the round
    private Map<String, ParticipantData> roundParticipants;

    // Logger
    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    // Event that checks if enough number of devices connected to current round
    private Cancellable checkReadyToRunLearning;

    // Coordinator actor
    private ActorRef coordinator;

    // Ticker actor
    private ActorRef tickActor;

    @Override
    public void onReceive(Object message) throws Exception {
        log.info("onReceive({})", message);
        Configuration.ConfigurationDTO configuration = Configuration.get();

        if (message instanceof StartRound) {
            // Message that round should start
            this.startRound();
        } else if (message instanceof InformAggregatorAboutNewParticipant) {
            // Message about new participant taking part in the new round
            InformAggregatorAboutNewParticipant messageCasted = (InformAggregatorAboutNewParticipant)message;
            ActorRef deviceReference = messageCasted.deviceReference;
            log.info("Path: " + deviceReference.path());
            this.roundParticipants.put(messageCasted.clientId,
                    new ParticipantData(deviceReference, messageCasted.address, messageCasted.port));
        } else if (message instanceof ReadyToRunLearningMessageResponse) {
            // Tell devices to run
            if (((ReadyToRunLearningMessageResponse) message).canStart) {
                this.checkReadyToRunLearning.cancel();
                for (ParticipantData participant : this.roundParticipants.values()) {
                    participant.deviceReference.tell(new StartLearningProcessCommand(configuration.modelConfig), getSelf());
                }
            }
        } else if (message instanceof StartLearningModule) {
            // Message when any of participants started their modules and server can start his own learning module
            // Updates corresponding device entity
            ActorRef sender = getSender();
            Optional<ParticipantData> first = roundParticipants.values().stream().findFirst();
            log.info("Sender: " + sender.path());
            log.info("First: " + first.get().deviceReference.path().toString());

            ParticipantData foundOnList = roundParticipants
                    .values()
                    .stream()
                    .filter(participantData -> participantData.deviceReference.equals(sender))
                    .findAny()
                    .orElse(null);

            foundOnList.moduleStarted = true;

            boolean allParticipantsStartedModule = roundParticipants
                    .values()
                    .stream()
                    .allMatch(participantData -> participantData.moduleStarted);

            log.info("Found on list" + (foundOnList != null));
            log.info("All participants started module" + allParticipantsStartedModule);

            if (allParticipantsStartedModule)
                for (ParticipantData participant : this.roundParticipants.values())
                    participant.deviceReference.tell(new AreYouAliveQuestion(), getSelf());
        } else if (message instanceof IAmAlive) {
            // Message sent at the beginning of learning, indicating that the sender is alive
            ActorRef sender = getSender();
            ParticipantData foundOnList = roundParticipants
                    .values()
                    .stream()
                    .filter(participantData -> participantData.deviceReference.equals(sender))
                    .findAny()
                    .orElse(null);

            foundOnList.moduleAlive = true;

            boolean allParticipantsAlive = roundParticipants
                    .values()
                    .stream()
                    .allMatch(participantData -> participantData.moduleAlive);

            if (allParticipantsAlive){
                this.exchange(configuration.minimumNumberOfDevices-1);
                this.runLearning();
                this.coordinator.tell(new RoundEnded(), getSelf());
            }
        } else {
            unhandled(message);
        }
    }

    private void exchange(int numberOfKeys) {
        int numberOfParticipants = roundParticipants.size();
        Map<String, String> addresses = roundParticipants
                .entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey,
                        participant -> participant.getValue().address));
        Map<String, Integer> ports = roundParticipants
                .entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey,
                        participant -> participant.getValue().port));
        List<Float> publicKeys = new ArrayList<>();
        Random keyGeneration = new Random();
        for (int k=0; k<numberOfKeys; k++) publicKeys.add(keyGeneration.nextFloat());

        for (ParticipantData participant : this.roundParticipants.values())
            participant.deviceReference.tell(new ClientDataSpread(numberOfParticipants, addresses, ports, publicKeys), getSelf());
    }

    // Stores information about each participant
    private static class ParticipantData {
        private ParticipantData(ActorRef deviceReference, /*String clientId, */String address, int port) {
            this.deviceReference = deviceReference;
            //this.clientId = clientId;
            this.moduleStarted = false;
            this.moduleAlive = false;
            this.port = port;
            this.address = address;
            this.interRes = new ArrayList<>();
        }

        //public String clientId;
        public ActorRef deviceReference;
        public boolean moduleStarted;
        public boolean moduleAlive;
        public int port;
        public String address;
        public List<Float> interRes;
    }

    // Starts new round
    private void startRound() {
        ActorSystem system = getContext().system();

        // Clears list of participants
        this.roundParticipants = new HashMap<>();
        // Cancels events from previous round
        if (this.checkReadyToRunLearning != null) {
            this.checkReadyToRunLearning.cancel();
            this.checkReadyToRunLearning = null;
        }

        // Event that checks if minimum participants joined current round
        FiniteDuration duration =  new FiniteDuration(60, TimeUnit.SECONDS);
        this.checkReadyToRunLearning = system
            .scheduler()
            .schedule(
                duration,
                duration,
                tickActor,
                new CheckReadyToRunLearningMessage(this.roundParticipants, getSelf()),
                system.dispatcher(),
                ActorRef.noSender());
    }

    // Starts server learning module
    private void runLearning() {
        System.out.println("Working Directory = " + System.getProperty("user.dir"));

        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.directory(new File(System.getProperty("user.dir")));

        Configuration.ConfigurationDTO configuration = Configuration.get();

        String participantsJson = getParticipantsJson();
        String tempvar = participantsJson.replace('"', '\'');
        // Executing module script as a command
        processBuilder
            .inheritIO()
            .command("python", configuration.serverModuleFilePath,
            "--datapath", configuration.testDataPath,
            "--participantsjsonlist", tempvar,
            "--epochs", String.valueOf(configuration.epochs),
            "--modelpath", configuration.savedModelPath,
            "--model_config", configuration.modelConfig,
            "--model_output", String.valueOf(configuration.targetOutputSize));

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
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    // TODO move to messages?
    public static class CheckReadyToRunLearningMessage {
        public Map<String, ParticipantData> participants;
        public ActorRef replayTo;
        public CheckReadyToRunLearningMessage(Map<String, ParticipantData> participants, ActorRef replayTo) {
            this.participants = participants;
            this.replayTo = replayTo;
        }
    }

    public static class ReadyToRunLearningMessageResponse {
        public Boolean canStart;
        public ReadyToRunLearningMessageResponse(Boolean canStart) {
            this.canStart = canStart;
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
