package example.akka.remote.client;

import akka.actor.*;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.Messages;
import scala.concurrent.duration.FiniteDuration;

import java.io.*;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

public class ClientActor extends UntypedActor {

    public ClientActor() {
        try {
            Configuration.ConfigurationDTO configuration = Configuration.get();
            this.address = configuration.address;
            this.pathToModules = configuration.pathToModules;
            this.port = configuration.port;
            this.clientId = configuration.id;

            // Getting the other actor   flserver.eastus.azurecontainer.io:5000
            this.selection = getContext().actorSelection("akka.tcp://AkkaRemoteServer@" + address + "/user/Selector");
            this.injector = getContext().actorSelection("akka.tcp://AkkaRemoteServer@" + address + "/user/Injector");
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    private String address;
    private String pathToModules;
    private int port;
    private String clientId;

    private ActorSelection selection;
    private ActorSelection injector;

    //     private ActorSelection selection = getContext().actorSelection("akka.tcp://AkkaRemoteServer@127.0.0.1:2552/user/CalculatorActor");

    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof Messages.StartLearning) {
            ActorSelection selection2 = getContext().actorSelection("akka.tcp://AkkaRemoteServer@" + address + "/user/Selector");
            log.info("injector -> " + injector.pathString());
            log.info("selection -> " + selection.pathString());
            log.info("injector -> " + "akka.tcp://AkkaRemoteServer@" + address + "/user/Injector");
            injector.tell(new Messages.GetModulesListRequest(((Messages.StartLearning) message).id), getSelf());
            log.info("After send to selector, address -> " + this.address);
        } else if(message instanceof Messages.GetModulesListResponse) {
            Messages.ModuleData module = this.findPropperModuleStrategy(((Messages.GetModulesListResponse) message).modules);
            File moduleFile = new File(pathToModules + module.fileName);
            if (!moduleFile.exists()) {
                selection.tell(new Messages.JoinRoundRequest(LocalDateTime.now(), 1, this.clientId, this.port), getSelf());
            }
            getSender().tell(new Messages.GetModuleRequest(module.fileName), getSelf());
        } else if (message instanceof Messages.GetModuleResponse) {
            Messages.GetModuleResponse result = (Messages.GetModuleResponse) message;
            log.info("File name: " + result.name + ", length: " + result.content.length);
            try (FileOutputStream fos = new FileOutputStream(pathToModules + result.name)) {
                fos.write(result.content);
            } catch (Exception e) {
                log.info("Error:");
                e.printStackTrace();
            }
            log.info("File saved");
            selection.tell(new Messages.JoinRoundRequest(LocalDateTime.now(), 1, this.clientId, this.port), getSelf());
        } else if (message instanceof Messages.JoinRoundResponse) {
            Messages.JoinRoundResponse result = (Messages.JoinRoundResponse) message;
            log.info("Got join round response {}", result.isLearningAvailable);
        } else if (message instanceof Messages.StartLearningProcessCommand) {
            log.info("Received start learning command");

            ActorSystem system = getContext().system();

            // Start learning module
            ActorRef moduleRummer = system.actorOf(Props.create(ClientRunModuleActor.class), "ClientRunModuleActor");
            moduleRummer.tell(new RunModule(), getSelf());

            ActorRef server = getSender();
            FiniteDuration delay =  new FiniteDuration(60, TimeUnit.SECONDS);

            // Tell server after 60 sec that script has been ran
            system
                .scheduler()
                .scheduleOnce(delay, server, new Messages.StartLearningModule(), system.dispatcher(), getSelf());
        }
    }

    private Messages.ModuleData findPropperModuleStrategy(List<Messages.ModuleData> modules) throws Exception {
        try {
            Configuration.ConfigurationDTO resourceinformation = Configuration.get();
            Messages.ModuleData m = modules.stream().findFirst().get();
            Optional<Messages.ModuleData> moduleOpt = modules
                    .stream()
                    .filter(element ->
                            element.useCUDA.equals(resourceinformation.useCuda)
                            && element.instanceType == resourceinformation.instanceType
                            && element.minRAMInGB <= resourceinformation.RAMInGB)
                    .findFirst();
            Messages.ModuleData module = moduleOpt.orElse(null);

            if (module == null) {
                throw new Exception("Could not find propper module");
            }
            return module;
        } catch (IOException e) {
            e.printStackTrace();
            throw e;
        }
    }

    public static class RunModule {
    }
}
