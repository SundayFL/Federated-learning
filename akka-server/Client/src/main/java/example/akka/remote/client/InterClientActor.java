package example.akka.remote.client;

import akka.actor.*;

public class InterClientActor extends UntypedActor {

    public InterClientActor(){

    }

    private ClientActor clientActor;
    private int numberOfClientstoAwait;
    private float[] rValues;
    private float[] own_rValues;

    @Override
    public void onReceive(Object message) throws Exception {

    }
}