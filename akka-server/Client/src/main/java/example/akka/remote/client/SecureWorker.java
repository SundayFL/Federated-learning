package example.akka.remote.client;

import akka.actor.UntypedActor;
import example.akka.remote.shared.Messages;

import java.io.File;
import java.io.IOException;

public class SecureWorker extends UntypedActor {

    public SecureWorker() { }

    @Override
    public void onReceive(Object message) throws Exception {
        if (message instanceof Messages.MakeChannel) this.makeChannel(((Messages.MakeChannel) message).participants);
    }

    private void makeChannel(String tempvar) throws IOException, InterruptedException {
        ProcessBuilder processBuilder = new ProcessBuilder();
        processBuilder.directory(new File(System.getProperty("user.dir")));
        Configuration configurationHandler = new Configuration();
        Configuration.ConfigurationDTO configuration = configurationHandler.get();

        processBuilder
                .inheritIO()
                .command("python", configuration.pathToChannel,
                        "--datapath", configuration.testdatapath,
                        "--participantsjsonlist", tempvar,
                        "--epochs", String.valueOf(configuration.epochs));

        Process process = processBuilder.start();
        int exitCode = process.waitFor();
    }
}
