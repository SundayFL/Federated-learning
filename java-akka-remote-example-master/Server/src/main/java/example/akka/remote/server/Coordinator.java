package example.akka.remote.server;

import akka.actor.ActorRef;
import akka.actor.Props;
import akka.actor.UntypedActor;
import akka.event.Logging;
import akka.event.LoggingAdapter;
import example.akka.remote.shared.LoggingActor;
import org.python.core.Options;

import javax.script.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.StringWriter;
import java.util.Arrays;
import java.util.List;

import static example.akka.remote.shared.Messages.*;

public class Coordinator extends UntypedActor {

    public Coordinator() {
        log.info("Constructor executed");
        this.selector = getContext().system().actorOf(Props.create(Selector.class), "Selector");
        // Start first round
        this.startRound();
    }

    private LoggingAdapter log = Logging.getLogger(getContext().system(), this);

    private ActorRef loggingActor = getContext().actorOf(Props.create(LoggingActor.class), "LoggingActor");

    private ActorRef selector;

    private ActorRef aggregator;

    @Override
    public void onReceive(Object message) throws Exception {
        log.info("onReceive({})", message);

        if (message instanceof JoinRoundRequest) {
            log.info("ERROR MESSAGE");
        } else {
            unhandled(message);
        }
    }

    private void startRound() {
        this.aggregator = getContext().system().actorOf(Props.create(Aggregator.class), "Aggregator");
        this.selector.tell(new StartRoundCoordinatorSelector(this.aggregator), getSelf());
    }

    private void RunPython() {
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

    public void printVariables() {
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
    }
}
