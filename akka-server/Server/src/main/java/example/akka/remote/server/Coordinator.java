package example.akka.remote.server;

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.LoggingActor;

import static example.akka.remote.shared.Messages.*;

public class Coordinator extends UntypedActor {

    public Coordinator() {
        this.selector = getContext().system().actorOf(Props.create(Selector.class), "Selector");
        this.aggregator = getContext().system().actorOf(Props.create(Aggregator.class, getSelf()), "Aggregator");
        // Start first round
        this.startRound();
    }

    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    private ActorRef loggingActor = getContext().actorOf(Props.create(LoggingActor.class), "LoggingActor");

    // Selector actor
    private ActorRef selector;

    // Aggregator actor
    private ActorRef aggregator;

    @Override
    public void onReceive(Object message) {
        log.info("onReceive({})", message);

        if (message instanceof RoundEnded) {
            log.info("Coordinator -> Received information that round has ended");
            // Starting new round
            startRound();
        } else {
            unhandled(message);
        }
    }

    private void startRound() {
        this.aggregator.tell(new StartRound(), getSelf());
        this.selector.tell(new StartRoundCoordinatorSelector(this.aggregator), getSelf());
    }
}
