package com.example;

import akka.actor.*;
import akka.actor.typed.ActorRef;
import akka.actor.typed.Behavior;
import akka.actor.typed.javadsl.*;

import akka.actor.ActorSelection;
import akka.actor.ActorSystem;

public class GreeterMain extends AbstractActor {

    public static class Start {
        public Start() {

        }
    }

    private class MessageResponse {
        private String getMessage() {
            return "Hello word";
        }
    }

    public static class SayHello {
        public final String name;

        public SayHello(String name) {
            this.name = name;
        }
    }

    public static Props props() {
        return Props.create(GreeterMain.class);

    }

    public static GreeterMain create() {
        return new GreeterMain();
    }

    public GreeterMain() {
    }

    @Override
    public Receive createReceive() {
        System.out.println("createReceive");
        return receiveBuilder().match(String.class, msg -> {
            System.out.println("Message sent:" + msg);
        }).match(Start.class, msg -> {
            System.out.println("Start");
            ActorSelection selection = getContext().system().actorSelection("akka://helloakka@127.0.0.1:5001/user/greeter");
            System.out.println("After selection");
            try {
                selection.tell(new SayHello("Hello server"), getSelf());

            } catch (Exception e) {
                System.out.println("Exception occurred");
            }
            System.out.println("After sent");
        }).build();
    }
}