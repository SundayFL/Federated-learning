package example.akka.remote.client;

import akka.actor.*;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.Messages;
import scala.concurrent.duration.FiniteDuration;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStreamReader;
import java.time.LocalDateTime;
import java.util.concurrent.TimeUnit;

public class ClientActor extends UntypedActor {
    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    // Getting the other actor   AkkaRemoteServer@flserver.eastus.azurecontainer.io:5000
    private ActorSelection selection = getContext().actorSelection("akka.tcp://AkkaRemoteServer@127.0.0.1:5000/user/Selector");

    @Override
    public void onReceive(Object message) throws Exception {
        if (message.equals("Start")) {
            selection.tell(new Messages.JoinRoundRequest(LocalDateTime.now(), 1, 1), getSelf());
            log.info("After send to selector");
        } else if (message instanceof Messages.JoinRoundResponse) {
            Messages.JoinRoundResponse result = (Messages.JoinRoundResponse) message;
            log.info("Got join round response {}", result.isLearningAvailable);
        } else if (message instanceof Messages.StartLearningProcessCommand) {
            log.info("Received start learning command");

            ActorSystem system = getContext().system();

            // Start learning module
            ActorRef moduleRummer = system.actorOf(Props.create(ClientRunModuleActor.class), "ClientRunModuleActor");
            moduleRummer.tell(new RunModule(), getSelf());

            ActorRef server = getSender();
            FiniteDuration delay =  new FiniteDuration(60, TimeUnit.SECONDS);

            // Tell server after 60 sec that script has been ran
            system
                .scheduler()
                .scheduleOnce(delay, server, new Messages.StartLearningModule(), system.dispatcher(), getSelf());
        }
    }

    public static class RunModule {
    }
}
