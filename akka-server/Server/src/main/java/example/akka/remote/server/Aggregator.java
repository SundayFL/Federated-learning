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
import scala.util.control.TailCalls;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.Duration;
import java.time.Instant;
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

    public Aggregator(){

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

    // Number of clients to await
    private int numberOfClientsToAwait;

    // Public keys
    private Map<String, Float> publics;

    // Time measurement
    Instant start, finish;

    // Enable secure aggregation
    boolean secureAgg = true;

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
                start = Instant.now();
                for (ParticipantData participant : this.roundParticipants.values()) {
                    participant.deviceReference.tell(new StartLearningProcessCommand(configuration.modelConfig), getSelf());
                }
            }
        } else if (message instanceof StartLearningModule) {
            this.numberOfClientsToAwait = roundParticipants.size();
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
                if (secureAgg) for (ParticipantData participant : this.roundParticipants.values())
                    participant.deviceReference.tell(new AreYouAliveQuestion(), getSelf());
                else {
                    log.info("Run learning");
                    this.runLearning();
                    log.info("Round ended");
                    finish = Instant.now();
                    float timeOfLearning = (float) (Duration.between(start, finish).toMillis()/1000.0);
                    log.info("Time of learning round: " + timeOfLearning);
                    this.coordinator.tell(new RoundEnded(), getSelf());
                }
        } else if (message instanceof IAmAlive) {
            log.info( "RECEIVED I AM ALIVED" );
            // Message sent at the beginning of learning, indicating that the sender is alive
            ActorRef sender = getSender();
            ParticipantData foundOnList = roundParticipants
                    .values()
                    .stream()
                    .filter(participantData -> participantData.deviceReference.equals(sender))
                    .findAny()
                    .orElse(null);

            if(foundOnList == null) return;
            foundOnList.moduleAlive = true;

            boolean allParticipantsAlive = roundParticipants
                    .values()
                    .stream()
                    .allMatch(participantData -> participantData.moduleAlive);

            log.info("Found on list {}", true);
            log.info("Everyone alive {}", allParticipantsAlive);

            if (allParticipantsAlive) { // everybody is alive
                log.info( "ALL PARTICIPANTS ALIVE" );
                log.info("Spreading data");
                this.exchange(configuration.minimumNumberOfDevices - 1);
                // spreading references to let clients exchange data
            }
        } else if (message instanceof SendInterRes) {
            // save InterRes
            // counting down clients to await InterRes values from
            this.numberOfClientsToAwait--;
            // how many InterRes values left
            if (numberOfClientsToAwait>0)
                log.info("Tensor received, "+numberOfClientsToAwait+" tensor"+(numberOfClientsToAwait==1?"":"s")+" left");
            else
                log.info("All tensors received");
            // retrieve the binary file sent
            byte[] bytes = ((SendInterRes) message).bytes;
            // and the sender's id
            String clientId = ((SendInterRes) message).sender;
            // save the binary file
            File dir = new File(configuration.pathToResources+"/interRes");
            boolean make;
            if (!dir.exists()) make = dir.mkdir();
            Files.write(Paths.get(configuration.pathToResources+"/interRes/"+clientId+".pt"), bytes);

            // when everybody has sent the weights
            if (numberOfClientsToAwait==0) {
                // Learning through deciphering learned models from an equation
                log.info("Run learning");
                this.runLearning();
                // round ends
                log.info("Round ended");
                finish = Instant.now();
                // elapsed time
                float timeOfLearning = (float) (Duration.between(start, finish).toMillis()/1000.0);
                log.info("Time of learning round: "+timeOfLearning);
                // tell the coordinator
                this.coordinator.tell(new RoundEnded(), getSelf());
            }
        } else {
            unhandled(message);
        }
    }

    public void exchange(int minimum) {
        int numberOfParticipants = roundParticipants.size();
        Random keyGeneration = new Random();

        // participants of the round with their devices' references and public keys
        // public keys are being freshly generated
        Map<String, Messages.ContactData> contactMap = roundParticipants
                .entrySet()
                .stream()
                .collect(Collectors.toMap(Map.Entry::getKey,
                        participant -> new Messages.ContactData(participant.getValue().deviceReference, keyGeneration.nextFloat())));

        // keep public keys in a map for later re-use
        this.publics = contactMap
                .entrySet()
                .stream()
                .collect(Collectors.toMap(participant -> '"'+participant.getKey()+'"', participant -> participant.getValue().publicKey));
        // spread the data
        for (Map.Entry<String, ParticipantData> participant : this.roundParticipants.entrySet())
            participant.getValue().deviceReference.tell(new ClientDataSpread(
                    participant.getKey(),
                    numberOfParticipants,
                    minimum,
                    contactMap
            ), getSelf());
    }

    // Stores information about each participant
    public static class ParticipantData {
        public ParticipantData(ActorRef deviceReference, String address, int port) {
            this.deviceReference = deviceReference;
            this.moduleStarted = false;
            this.moduleAlive = false;
            this.port = port;
            this.address = address;
            this.interRes = new ArrayList<>();
        }

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
        FiniteDuration duration =  new FiniteDuration(10, TimeUnit.SECONDS);
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
            .command("python", configuration.secureAgg?configuration.serverModuleFilePathSA:configuration.serverModuleFilePath,
            "--datapath", configuration.testDataPath,
            "--participantsjsonlist", tempvar,
            "--publicKeys", configuration.secureAgg?this.publics.toString():"",
            "--degree", String.valueOf(this.roundParticipants.size()),
            "--epochs", String.valueOf(configuration.epochs),
            "--modelpath", configuration.secureAgg?configuration.savedModelPathSA:configuration.savedModelPath,
            "--pathToResources", configuration.pathToResources,
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



    // GETTERS & SETTERS


    public Map<String, ParticipantData> getRoundParticipants() {
        return roundParticipants;
    }

    public void setRoundParticipants(Map<String, ParticipantData> roundParticipants) {
        this.roundParticipants = roundParticipants;
    }

    public LoggingAdapter getLog() {
        return log;
    }

    public void setLog(LoggingAdapter log) {
        this.log = log;
    }

    public Cancellable getCheckReadyToRunLearning() {
        return checkReadyToRunLearning;
    }

    public void setCheckReadyToRunLearning(Cancellable checkReadyToRunLearning) {
        this.checkReadyToRunLearning = checkReadyToRunLearning;
    }

    public ActorRef getCoordinator() {
        return coordinator;
    }

    public void setCoordinator(ActorRef coordinator) {
        this.coordinator = coordinator;
    }

    public ActorRef getTickActor() {
        return tickActor;
    }

    public void setTickActor(ActorRef tickActor) {
        this.tickActor = tickActor;
    }

    public int getNumberOfClientsToAwait() {
        return numberOfClientsToAwait;
    }

    public void setNumberOfClientsToAwait(int numberOfClientsToAwait) {
        this.numberOfClientsToAwait = numberOfClientsToAwait;
    }

    public Map<String, Float> getPublics() {
        return publics;
    }

    public void setPublics(Map<String, Float> publics) {
        this.publics = publics;
    }

    public Instant getStart() {
        return start;
    }

    public void setStart(Instant start) {
        this.start = start;
    }

    public Instant getFinish() {
        return finish;
    }

    public void setFinish(Instant finish) {
        this.finish = finish;
    }

    public boolean isSecureAgg() {
        return secureAgg;
    }

    public void setSecureAgg(boolean secureAgg) {
        this.secureAgg = secureAgg;
    }
}
