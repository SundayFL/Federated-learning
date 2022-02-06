package example.akka.remote.server;

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.LoggingActor;
import org.python.core.Options;

import javax.script.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.StringWriter;
import java.util.List;

import static example.akka.remote.shared.Messages.*;

public class Selector extends UntypedActor {

    public Selector() {
        log.info("Selector created");

    }

    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    private ActorRef loggingActor = getContext().actorOf(Props.create(LoggingActor.class), "LoggingActor");

    // Flag that tels if currently is running a round
    private boolean isRoundActive = false;

    // Reference to aggregator actor
    private ActorRef aggregator;

    @Override
    public void onReceive(Object message) throws Exception {
        log.info("onReceive({})", message);

        if (message instanceof JoinRoundRequest) {
            // Receives join request from the device
            log.info("Selector received join request");
            ActorRef deviceReference = getSender();
            log.info("Selector path: " + deviceReference.path());
            int port = ((JoinRoundRequest) message).port;
            String clientId = ((JoinRoundRequest) message).clientId;
            String address = ((JoinRoundRequest) message).address;
            deviceReference.tell(new JoinRoundResponse(this.isRoundActive, this.aggregator), getSelf());

            // tell aggregator about new device
            this.aggregator.tell(new InformAggregatorAboutNewParticipant(deviceReference, clientId, address, port), getSelf());
        } else if (message instanceof StartRoundCoordinatorSelector) {
            this.isRoundActive = true;
            this.aggregator = ((StartRoundCoordinatorSelector) message).aggregator;
        } else {
            unhandled(message);
        }
    }
}
