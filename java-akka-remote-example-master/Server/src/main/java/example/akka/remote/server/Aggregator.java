package example.akka.remote.server;

import akka.actor.*;
import akka.actor.dsl.Creators;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.remote.transport.ThrottlerTransportAdapter;
import example.akka.remote.shared.LoggingActor;
import example.akka.remote.shared.Messages;
import scala.concurrent.duration.FiniteDuration;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

import static example.akka.remote.shared.Messages.*;

public class Aggregator extends UntypedActor {

    public Aggregator() {
        log.info("Selector created");
        this.startRound();
    }

    private List<ParticipantData> roundParticipants;

    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    private ActorRef loggingActor = getContext().actorOf(Props.create(LoggingActor.class), "LoggingActor");

    private Cancellable checkReadyToRunLearning;

    @Override
    public void onReceive(Object message) throws Exception {
        log.info("onReceive({})", message);

        if (message instanceof InformAggregatorAboutNewParticipant) {
            ActorRef deviceReference = ((InformAggregatorAboutNewParticipant) message).deviceReference;
            log.info("Path: " + deviceReference.path());
            this.roundParticipants.add(new ParticipantData(deviceReference));
        } else if (message instanceof ReadyToRunLearningMessageResponse) {
            // tell devices to run
            if (((ReadyToRunLearningMessageResponse) message).canStart) {
                this.checkReadyToRunLearning.cancel();
                for (ParticipantData participant : this.roundParticipants) {
                    participant.deviceReference.tell(new StartLearningProcessCommand(), getSelf());
                }
            }
        } else if (message instanceof StartLearningModule) {
            ActorRef sender = getSender();
            Optional<ParticipantData> first = roundParticipants.stream().findFirst();
            log.info("Sender: " + sender.path());
            log.info("First: " + first.get().deviceReference.path().toString());

            ParticipantData foundOnList = roundParticipants
                    .stream()
                    .filter(participantData -> participantData.deviceReference.equals(sender))
                    .findAny()
                    .orElse(null);

            foundOnList.moduleStarted = true;

            boolean allParticipantsStartedModule = roundParticipants
                    .stream()
                    .allMatch(participantData -> participantData.moduleStarted);

            log.info("Found on list" + (foundOnList != null));
            log.info("All participants started module" + allParticipantsStartedModule);

            if (allParticipantsStartedModule){
                this.runLearning2();
            }
        } else {
            unhandled(message);
        }
    }

    private static class ParticipantData {
        public ParticipantData(ActorRef deviceReference) {
            this.deviceReference = deviceReference;
            this.moduleStarted = false;
        }

        public ActorRef deviceReference;
        public boolean moduleStarted;
    }

    private void startRound() {
        ActorSystem system = getContext().system();

        // system.actorOf(Props.create(Aggregator.class), "CalculatorActor");
        this.roundParticipants = new ArrayList<>();

        ActorRef tickActor = system.actorOf(Props.create(Ticker.class), "Ticker");

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

    public static class CheckReadyToRunLearningMessage {
        public List<ParticipantData> participants;
        public ActorRef replayTo;
        public CheckReadyToRunLearningMessage(List<ParticipantData> participants, ActorRef replayTo) {
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

    private void runLearning2() {
        System.out.println("Working Directory = " + System.getProperty("user.dir"));

        ProcessBuilder processBuilder2 = new ProcessBuilder();
        processBuilder2.directory(new File(System.getProperty("user.dir")));
        System.out.println("Check python version");
        processBuilder2.inheritIO().command("python", "--version");
        processBuilder2.inheritIO().command("ls", "src/main/python");
        try {
            Process process2 = processBuilder2.start();
            int exitCode = process2.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
        System.out.println("After ls");
        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.directory(new File(System.getProperty("user.dir")));

        processBuilder.inheritIO().command("python", "./src/main/python/client.py", "--datapath", "./src/main/python/data",
                "--participantsjsonlist", "{\"id\": \"alice\", \"port\": \"8777\"}", "--epochs", "10", "--modelpath",
                "./saved_model");
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
