package example.akka.loggerenvironment;

import akka.actor.ActorSystem;
import com.typesafe.config.ConfigFactory;

public class LoggerEnvironment {
    public static void main(String[] args) {
        // Creating environment
        ActorSystem.create("AkkaLoggerSystem", ConfigFactory.load());
    }
}
