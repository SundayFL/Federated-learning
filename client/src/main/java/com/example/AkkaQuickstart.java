package com.example;

import akka.actor.AbstractActor;
import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;

import java.io.IOException;
public class AkkaQuickstart {
  public static void main(String[] args) {
    //#actor-system
    //final AbstractActor greeterMain = GreeterMain.create();
    Config config = ConfigFactory.load();

    ActorSystem system = ActorSystem.create("SwapperSystem", config);
    ActorRef mainActor = system.actorOf(Props.create(GreeterMain.class));

    mainActor.tell(new GreeterMain.Start(), ActorRef.noSender());
    //Props props1 = Props.create(GreeterMain.class);

    //#actor-system

    //#main-send-messages
    //greeterMain..tell(new GreeterMain.Start());
    //#main-send-messages

    try {
      System.out.println(">>> Press ENTER to exit <<<");
      System.in.read();
    }
    catch (IOException ignored) {

    } finally {
      //greeterMain.terminate();
    }
  }
}
