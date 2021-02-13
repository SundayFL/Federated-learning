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

    private boolean isRoundActive = false;

    private ActorRef aggregator;

    @Override
    public void onReceive(Object message) throws Exception {
        log.info("onReceive({})", message);

        if (message instanceof JoinRoundRequest) {
            log.info("Selector received join request");
            ActorRef deviceReference = getSender();
            log.info("Selector path: " + deviceReference.path());
            deviceReference.tell(new JoinRoundResponse(this.isRoundActive, this.aggregator), getSelf());

            // tell aggregator about new device
            this.aggregator.tell(new InformAggregatorAboutNewParticipant(deviceReference), getSelf());
        } else if (message instanceof StartRoundCoordinatorSelector) {
            this.isRoundActive = true;
            this.aggregator = ((StartRoundCoordinatorSelector) message).aggregator;
        } else {
            unhandled(message);
        }
    }
}
