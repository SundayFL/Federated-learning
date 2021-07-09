package example.akka.remote.server;

import akka.actor.ActorContext;
import akka.actor.ActorRef;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.LoggingActor;

import javax.script.*;
import java.io.*;
import java.util.List;
import org.python.core.Options;

import static example.akka.remote.shared.Messages.*;

public class CalculatorActor extends UntypedActor {

    public CalculatorActor() {
        log.info("Constructor executed - ERROR");
        //getContext().system().actorOf(Props.create(CalculatorActor.class), "CalculatorActor");
    }

    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    // private ActorRef loggingActor = getContext().actorOf(Props.create(LoggingActor.class), "LoggingActor");

    @Override
    public void onReceive(Object message) throws Exception {
        log.info("ERROR", message);

        /*if (message instanceof Sum) {
            log.info("ERROR MESSAGE");
            Sum sum = (Sum) message;
            int result = sum.getFirst() + sum.getSecond();
            getSender().tell(new Result(result), getSelf());

            loggingActor.tell(sum.getFirst() + " + " + sum.getSecond() + " = " + result, getSelf());
        } else if (message instanceof JoinRoundRequest) {
            RunPython();

            log.info("JoinRoundRequest MESSAGE");
            getSender().tell(new JoinRoundResponse(true), getSelf());
        } else {
            unhandled(message);
        }*/
    }

    /*private void runDocker() throws IOException {
        Process p = Runtime.getRuntime().exec("docker images");

        DockerClientConfig standard = DefaultDockerClientConfig.createDefaultConfigBuilder().build();

        DockerClientConfig custom = DefaultDockerClientConfig.createDefaultConfigBuilder()
                .withDockerHost("tcp://docker.somewhere.tld:2376")
                .withDockerTlsVerify(true)
                .withDockerCertPath("/home/user/.docker")
                /*.withRegistryUsername(registryUser)
                .withRegistryPassword(registryPass)
                .withRegistryEmail(registryMail)
                .withRegistryUrl(registryUrl)*
                .build();

        DockerHttpClient httpClient = new ApacheDockerHttpClient.Builder()
                .dockerHost(standard.getDockerHost())
                .sslConfig(standard.getSSLConfig())
                .build();
    }*/

    /*private void RunPython() {
        printVariables();

        StringWriter writer = new StringWriter();
        ScriptContext context = new SimpleScriptContext();
        context.setWriter(writer);
        Options.importSite = false;
        ScriptEngineManager manager = new ScriptEngineManager();
        ScriptEngine engine = manager.getEngineByName("python");
        ScriptEngine engine27 = manager.getEngineByName("client.cpython-37.pyc");

        try {
            String path = resolvePythonScriptPath("hello.py");
            FileReader fileReader = new FileReader(path);
            if (engine == null) {
                log.info("engine is null | SO without null");
            }
            if (engine27 == null) {
                log.info("engine27≠ is null | SO without null");
            }
            if (context == null) {
                log.info("context is null");
            }
            if (path == null) {
                log.info("path is null");
            }
            if (fileReader == null) {
                log.info("fileReader is null");
            }
            engine.eval(fileReader, context);
        } catch (FileNotFoundException e) {
            log.info("File not found");
        } catch (Exception e) {
            log.info("Another exception - pyc: " + e);
        }

        String output = writer.toString().trim();
        log.info("Output: " + output);
    }

    private static String resolvePythonScriptPath(String filename) {
        File file = new File("src/main/python/" + filename);
        return file.getAbsolutePath();
    }

    public void printVariables()
    {
        ScriptEngineManager mgr = new ScriptEngineManager();
        List<ScriptEngineFactory> factories = mgr.getEngineFactories();
        for (ScriptEngineFactory factory : factories)
        {
            log.info("ScriptEngineFactory Info");
            String engName = factory.getEngineName();
            String engVersion = factory.getEngineVersion();
            String langName = factory.getLanguageName();
            String langVersion = factory.getLanguageVersion();
            log.info("\tScript Engine: " + engName + engVersion);
            List<String> engNames = factory.getNames();
            for (String name : engNames)
            {
                log.info("\tEngine Alias: " + name);
            }
            log.info("\tLanguage: " + langName + langVersion);
        }
    }*/
}
