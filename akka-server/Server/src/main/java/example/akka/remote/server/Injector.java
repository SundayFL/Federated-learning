package example.akka.remote.server;

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.LoggingActor;
import example.akka.remote.shared.Messages;

import java.io.File;
import java.io.FileInputStream;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

import static example.akka.remote.shared.Messages.*;

public class Injector extends UntypedActor {

    public Injector() {
        log.info("Injector created " + getSelf().path());
    }

    // List of modules
    // TODO should be read from json
    private List<Messages.ModuleData> modules = new ArrayList() {{
        add(new Messages.ModuleData("mnist","server.py", "Server", false, 4, InstanceType.Computer));
    }};

    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    @Override
    public void onReceive(Object message) throws Exception {
        log.info("onReceive({})", message);

        if (message instanceof GetModulesListRequest) {
            // Returns modules list
            List<Messages.ModuleData> filteredModules = modules
                    .stream()
                    .filter(x -> x.id.equals(((GetModulesListRequest) message).id))
                    .collect(Collectors.toList());
            getSender().tell(new GetModulesListResponse(filteredModules), getSelf());
        } else if (message instanceof GetModuleRequest) {
            // Returns module asked by device, reads it end returns content
            String name = ((GetModuleRequest) message).name;
            log.info("Searching for file: {}", name);
            byte[] bytes = Files.readAllBytes(Paths.get("./src/main/modules/learning/" + name));

            getSender().tell(new GetModuleResponse(bytes, name), getSelf());
        } else {
            unhandled(message);
        }
    }
}
