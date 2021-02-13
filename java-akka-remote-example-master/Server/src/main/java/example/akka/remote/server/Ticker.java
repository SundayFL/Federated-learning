package example.akka.remote.server;

import akka.actor.AbstractActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import akka.japi.pf.ReceiveBuilder;

class Ticker extends AbstractActor {

    private final LoggingAdapter log = Logging.getLogger(context().system(), this);

    public Ticker() {
        receive(ReceiveBuilder.
                match(Aggregator.CheckReadyToRunLearningMessage.class, s -> {
                    log.info("Received CheckReadyToRunLearningMessage message");
                    log.info("Ticker: numberOfDevices: " + s.participants.size());
                    s.replayTo.tell(new Aggregator.ReadyToRunLearningMessageResponse(s.participants.size() > 0), self());
                }).
                matchAny(o -> log.info("received unknown message")).build()
        );
    }
}