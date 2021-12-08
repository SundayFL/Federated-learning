package example.akka.remote.client;

import akka.actor.*;
import akka.actor.dsl.Creators;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.Messages;
import scala.concurrent.duration.FiniteDuration;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.TimeUnit;

public class ClientActor extends UntypedActor {

    public ClientActor() {
        try {
            // Getting configuration
            Configuration configurationHandler = new Configuration();
            Configuration.ConfigurationDTO configuration = configurationHandler.get();

            // Setting configuration
            this.address = configuration.address;
            this.pathToModules = configuration.pathToModules;
            this.port = configuration.port;
            this.clientId = configuration.id;

            // Getting the other actors
            // // flserver.eastus.azurecontainer.io:5000 - azure address
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
    private String modelConfig;

    private ActorSelection selection;
    private ActorSelection injector;
    private ActorRef server;

    private Map<String, Messages.ContactData> contactMap;
    private int numberOfClientstoAwait;

    @Override
    public void onReceive(Object message) throws Exception {
        // Message received at the beginning from main class
        if (message instanceof Messages.StartLearning) {
            this.taskId = ((Messages.StartLearning) message).id;

            // Finding proper module for specified task id
            List<ModulesManager.ModuleDTO> modules = ModulesManager.GetAvailableModules();

            ModulesManager.ModuleDTO module = modules
                    .stream()
                    .filter(x -> x.taskId.equals(this.taskId))
                    .findFirst()
                    .orElse(null);

            if (module == null) {
                // if module not found then ask for modules list
                injector.tell(new Messages.GetModulesListRequest(((Messages.StartLearning) message).id), getSelf());
                return;
            }
            // Set module filename
            this.moduleFileName = module.fileName;
            // When we confirm that we have module we can ask server to join round
            selection.tell(new Messages.JoinRoundRequest(LocalDateTime.now(), this.taskId, this.clientId, this.address, this.port), getSelf());
            log.info("After send to selector, address -> " + this.address);
        } else if(message instanceof Messages.GetModulesListResponse) {
            // Find the best module
            Messages.ModuleData module = this.findProperModuleStrategy(((Messages.GetModulesListResponse) message).modules);
            // Ask for module
            getSender().tell(new Messages.GetModuleRequest(module.fileName), getSelf());
        } else if (message instanceof Messages.GetModuleResponse) {
            // Save received module
            Messages.GetModuleResponse module = (Messages.GetModuleResponse) message;
            log.info("File name: " + module.fileName + ", length: " + module.content.length);
            SaveFile(module);
            log.info("File saved");
            ModulesManager.SaveModule(this.taskId, module.fileName);
            log.info("Module list saved");
            this.moduleFileName = module.fileName;
            selection.tell(new Messages.JoinRoundRequest(LocalDateTime.now(), this.taskId, this.clientId, this.address, this.port), getSelf());
        } else if (message instanceof Messages.JoinRoundResponse) {
            // Response if device can join round
            Messages.JoinRoundResponse result = (Messages.JoinRoundResponse) message;
            log.info("Got join round response {}", result.isLearningAvailable);
            // TODO Need to be handled negative scenario
        } else if (message instanceof Messages.StartLearningProcessCommand) {
            // Server told that device should run learning module
            log.info("Received start learning command");

            ActorSystem system = getContext().system();
            Messages.StartLearningProcessCommand messageWithModel = (Messages.StartLearningProcessCommand) message;
            this.modelConfig = messageWithModel.getModelConfig();

            // Start learning module
            ActorRef moduleRunner = system.actorOf(Props.create(ClientRunModuleActor.class), "ClientRunModuleActor");
            moduleRunner.tell(new RunModule(this.moduleFileName, this.modelConfig), getSelf());

            ActorRef server = getSender();
            FiniteDuration delay =  new FiniteDuration(10, TimeUnit.SECONDS);

            // Tell server, after 60 sec, that script has been run
            system
                .scheduler()
                .scheduleOnce(delay, server, new Messages.StartLearningModule(), system.dispatcher(), getSelf());
        } else if (message instanceof Messages.AreYouAliveQuestion){
            log.info("I am alive!");
            this.server = getSender(); // from here we save the server reference
            this.server.tell(new Messages.IAmAlive(), getSelf());
        } else if (message instanceof Messages.ClientDataSpread){
            this.numberOfClientstoAwait = ((Messages.ClientDataSpread) message).numberOfClients;
            this.contactMap = ((Messages.ClientDataSpread) message).contactMap;

            ActorSystem system = getContext().system();
            // Start reading R values
            ActorRef modelReader = system.actorOf(Props.create(ClientGetModelActor.class), "ClientGetModelActor");
            modelReader.tell(message, getSelf());
        } else if (message instanceof Messages.RValuesReady){
            // sending R values to other clients
            Configuration.ConfigurationDTO configuration;
            Configuration configurationHandler = new Configuration();
            configuration = configurationHandler.get();
            byte[] bytes;
            for (Map.Entry<String, Messages.ContactData> client: this.contactMap.entrySet()) {
                log.info("Sending R value to "+client.getKey());
                bytes = Files.readAllBytes(Paths.get(configuration.pathToResources+this.clientId+"/"+this.clientId+"_"+client.getKey()+".pt"));
                client.getValue().reference.tell(new Messages.SendRValue(this.clientId, bytes), getSelf());
            }
        } else if (message instanceof Messages.SendRValue){
            Configuration.ConfigurationDTO configuration;
            Configuration configurationHandler = new Configuration();
            configuration = configurationHandler.get();
            numberOfClientstoAwait--;
            log.info("Received R value from "+((Messages.SendRValue) message).sender);
            log.info("R values left: "+numberOfClientstoAwait);
            byte[] bytes = ((Messages.SendRValue) message).bytes;
            Files.write(Paths.get(configuration.pathToResources+this.clientId+"/"+((Messages.SendRValue) message).sender+"_"+this.clientId+".pt"), bytes);

            if (numberOfClientstoAwait==0){
                this.calculateInterRes();
                byte[] bytes2 = Files.readAllBytes(Paths.get(configuration.pathToResources+this.clientId+"/interRes.pt"));
                this.server.tell(new Messages.SendInterRes(this.clientId, bytes2), getSelf());
                log.info("InterRes sent");
            }
        }
    }

    // Saves file - module
    public void SaveFile(Messages.GetModuleResponse result) {
        try (FileOutputStream fos = new FileOutputStream(pathToModules + result.fileName)) {
            fos.write(result.content);
        } catch (Exception e) {
            log.info("Error:");
            e.printStackTrace();
        }
    }

    public void calculateInterRes(){
        log.info("Calculation");
        Configuration.ConfigurationDTO configuration;
        try {
            Configuration configurationHandler = new Configuration();
            configuration = configurationHandler.get();

            // execute scripts with proper parameters
            ProcessBuilder processBuilder = new ProcessBuilder();
            processBuilder.directory(new File(System.getProperty("user.dir")));
            log.info(configuration.pathToInterRes);
            processBuilder
                    .inheritIO()
                    .command("python", configuration.pathToInterRes,
                            "--pathToResources", configuration.pathToResources,
                            "--id", configuration.id);

            Process process = processBuilder.start();
            int exitCode = process.waitFor();
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    // Finds module that meets requirements
    public Messages.ModuleData findProperModuleStrategy(List<Messages.ModuleData> modules) throws Exception {
        try {
            Configuration configurationHandler = new Configuration();
            Configuration.ConfigurationDTO configuration = configurationHandler.get();

            Messages.ModuleData m = modules.stream().findFirst().get();
            Optional<Messages.ModuleData> moduleOpt = modules
                    .stream()
                    .filter(element ->
                            element.useCUDA.equals(configuration.useCuda)
                            && element.instanceType == configuration.instanceType
                            && element.minRAMInGB <= configuration.RAMInGB)
                    .findFirst();
            Messages.ModuleData module = moduleOpt.orElse(null);

            if (module == null) {
                throw new Exception("Could not find proper module");
            }
            return module;
        } catch (IOException e) {
            e.printStackTrace();
            throw e;
        }
    }

    // Run module message
    // TODO should be moved to messages
    public static class RunModule {
        public RunModule(String moduleFileName, String modelConfig) {
            this.moduleFileName = moduleFileName;
            this.modelConfig = modelConfig;
        }
        public String moduleFileName;
        public String modelConfig;
    }
}
