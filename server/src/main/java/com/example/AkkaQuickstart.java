package com.example;

import akka.actor.typed.ActorSystem;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;

import java.io.IOException;
public class AkkaQuickstart {

  public static void main(String[] args) {
    //#actor-system

    //actorOf(Props(...), actorName)

    Config config = ConfigFactory.load();

    final ActorSystem<GreeterMain.SayHello> greeterMain = ActorSystem.create(GreeterMain.create(),
            "helloakka",
            config);
    //#actor-system

    //#main-send-messages
    greeterMain.tell(new GreeterMain.SayHello("Charles"));
    //#main-send-messages

    try {
      System.out.println(">>> Press ENTER to exit <<<");
      System.in.read();
    }
    catch (IOException ignored) {

    } finally {
      greeterMain.terminate();
    }
  }
}
