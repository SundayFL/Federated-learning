package example.akka.remote.client;

import akka.actor.ActorRef;
import akka.actor.ActorSystem;
import akka.actor.Props;
import com.typesafe.config.Config;
import com.typesafe.config.ConfigFactory;
import example.akka.remote.shared.Messages;

import java.util.Properties;

public class Client {
    public static void main(String[] args) {

        SaveArguments(args);

        GetConfig(args);

        // Creating environment
        ActorSystem system = ActorSystem.create("AkkaRemoteClient", GetConfig(args));

        // Client actor
        ActorRef client = system.actorOf(Props.create(ClientActor.class));

        // Send a Calc job
        client.tell(new Messages.StartLearning("mnist"), ActorRef.noSender());
    }

    private static Config GetConfig(String[] args) {
        if (args.length > 0) {
            Properties properties = new Properties();
            properties.setProperty("akka.remote.netty.tcp.port", args[0]);
            Config overrides = ConfigFactory.parseProperties(properties);
            Config actualConfig = overrides.withFallback(ConfigFactory.load());
            return actualConfig;
        }
        return ConfigFactory.load();
    }

    private static void SaveArguments(String[] args) {
        Configuration configuration = new Configuration();
        configuration.SaveArguments(args);
    }
}
