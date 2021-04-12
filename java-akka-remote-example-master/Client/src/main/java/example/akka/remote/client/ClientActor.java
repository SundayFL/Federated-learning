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
    private String taskId;
    private String moduleFileName;

    private ActorSelection selection;
    private ActorSelection injector;

    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof Messages.StartLearning) {
            this.taskId = ((Messages.StartLearning) message).id;

            List<ModulesManager.ModuleDTO> modules = ModulesManager.GetAvailableModules();

            ModulesManager.ModuleDTO module = modules
                    .stream()
                    .filter(x -> x.taskId.equals(this.taskId))
                    .findFirst()
                    .orElse(null);

            if (module == null) {
                injector.tell(new Messages.GetModulesListRequest(((Messages.StartLearning) message).id), getSelf());
                return;
            }
            this.moduleFileName = module.fileName;
            selection.tell(new Messages.JoinRoundRequest(LocalDateTime.now(), this.taskId, this.clientId, this.port), getSelf());
            log.info("After send to selector, address -> " + this.address);
        } else if(message instanceof Messages.GetModulesListResponse) {
            Messages.ModuleData module = this.findProperModuleStrategy(((Messages.GetModulesListResponse) message).modules);
            getSender().tell(new Messages.GetModuleRequest(module.fileName), getSelf());
        } else if (message instanceof Messages.GetModuleResponse) {
            Messages.GetModuleResponse module = (Messages.GetModuleResponse) message;
            log.info("File name: " + module.fileName + ", length: " + module.content.length);
            SaveFile(module);
            log.info("File saved");
            ModulesManager.SaveModule(this.taskId, module.fileName);
            log.info("Module list saved");
            this.moduleFileName = module.fileName;
            selection.tell(new Messages.JoinRoundRequest(LocalDateTime.now(), this.taskId, this.clientId, this.port), getSelf());
        } else if (message instanceof Messages.JoinRoundResponse) {
            Messages.JoinRoundResponse result = (Messages.JoinRoundResponse) message;
            log.info("Got join round response {}", result.isLearningAvailable);
        } else if (message instanceof Messages.StartLearningProcessCommand) {
            log.info("Received start learning command");

            ActorSystem system = getContext().system();

            // Start learning module
            ActorRef moduleRummer = system.actorOf(Props.create(ClientRunModuleActor.class), "ClientRunModuleActor");
            moduleRummer.tell(new RunModule(this.moduleFileName), getSelf());

            ActorRef server = getSender();
            FiniteDuration delay =  new FiniteDuration(60, TimeUnit.SECONDS);

            // Tell server after 60 sec that script has been ran
            system
                .scheduler()
                .scheduleOnce(delay, server, new Messages.StartLearningModule(), system.dispatcher(), getSelf());
        }
    }

    private void SaveFile(Messages.GetModuleResponse result) {
        try (FileOutputStream fos = new FileOutputStream(pathToModules + result.fileName)) {
            fos.write(result.content);
        } catch (Exception e) {
            log.info("Error:");
            e.printStackTrace();
        }
    }

    private Messages.ModuleData findProperModuleStrategy(List<Messages.ModuleData> modules) throws Exception {
        try {
            Configuration.ConfigurationDTO resourceInformation = Configuration.get();
            Messages.ModuleData m = modules.stream().findFirst().get();
            Optional<Messages.ModuleData> moduleOpt = modules
                    .stream()
                    .filter(element ->
                            element.useCUDA.equals(resourceInformation.useCuda)
                            && element.instanceType == resourceInformation.instanceType
                            && element.minRAMInGB <= resourceInformation.RAMInGB)
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
        public RunModule(String moduleFileName) {
            this.moduleFileName = moduleFileName;
        }
        public String moduleFileName;
    }
}
