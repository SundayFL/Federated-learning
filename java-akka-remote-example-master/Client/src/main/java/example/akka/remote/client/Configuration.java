package example.akka.remote.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import example.akka.remote.shared.Messages;

import java.io.File;
import java.io.IOException;

public class Configuration {

    public static ConfigurationDTO get() throws IOException {
        ObjectMapper mapper = new ObjectMapper();

        Configuration.ConfigurationDTO configuration = mapper.readValue(new File("./src/main/resources/appConfig.json"), Configuration.ConfigurationDTO.class);
        return configuration;
    }

    public static class ConfigurationDTO {
        public Boolean useCuda;
        public int RAMInGB;
        public Messages.InstanceType instanceType;

        public String datapath;
        public String id;
        public String host;
        public int port;
        public String address;
        public String pathToModules;
    }
}


