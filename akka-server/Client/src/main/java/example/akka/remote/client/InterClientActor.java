package example.akka.remote.client;

import akka.actor.*;
import example.akka.remote.shared.Messages;

import java.util.List;
import java.util.Map;

public class InterClientActor extends UntypedActor {

    public InterClientActor(ActorRef clientActor){
        this.clientActor = clientActor;
    }

    private ActorRef clientActor;
    private int numberOfClientstoAwait;
    private Map<String, Float> RValues;
    private List<Float> ownRValues;

    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof Messages.ClientDataSpread) {
            this.numberOfClientstoAwait = ((Messages.ClientDataSpread) message).numberOfClients;
            // TODO save clients' addresses and ports, public keys in temporary JSON files
        }
    }
}