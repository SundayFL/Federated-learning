package example.akka.remote.client;

import akka.actor.*;
import com.fasterxml.jackson.databind.ObjectMapper;
import example.akka.remote.shared.Messages;

import java.io.File;
import java.util.List;
import java.util.Map;
import java.util.HashMap;

public class InterClientActor extends UntypedActor {

    public InterClientActor(String clientId, ActorRef clientActor){
        this.clientId = clientId;
        this.clientActor = clientActor;
    }

    private String clientId;
    private ActorRef clientActor;
    private int numberOfClientstoAwait;
    private Map<String, Float> RValues;
    private List<Float> ownRValues;

    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof Messages.ClientDataSpread) {
            this.numberOfClientstoAwait = ((Messages.ClientDataSpread) message).numberOfClients;
            ObjectMapper mapper = new ObjectMapper();
            Map<String, Object> JSONmap = new HashMap<>();
            JSONmap.put("addresses", ((Messages.ClientDataSpread) message).addresses);
            JSONmap.put("ports", ((Messages.ClientDataSpread) message).ports);
            JSONmap.put("publicKeys", ((Messages.ClientDataSpread) message).publicKeys);
            mapper.writeValue(new File("./src/main/resources/"+clientId+".json"), JSONmap);
        }
    }
}