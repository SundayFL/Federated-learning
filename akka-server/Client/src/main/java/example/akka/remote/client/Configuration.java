package example.akka.remote.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import example.akka.remote.shared.Messages;

import java.io.File;
import java.io.IOException;

public class Configuration {

    public static Integer dataSetId;  // number of dataset
    public static String id; // Id of the client e.g. alice
    public static Integer port; // port on which the client is working

    // Method which saves arguments passed as execution arguments
    public void SaveArguments(String[] args) {
        System.out.println("Len: " + args.length);

        if (args.length > 1) {
            this.dataSetId = Integer.parseInt(args[1]);
        }

        if (args.length > 2) {
            this.id = args[2];
        }
        if (args.length > 3) {
            this.port = Integer.parseInt(args[3]);
        }
        System.out.println("dataSetId: " + this.dataSetId + ", id: " + this.id + ", port: " + this.port);
    }

    // Method which returns configuration from appConfig.json file
    public ConfigurationDTO get() throws IOException {
        ObjectMapper mapper = new ObjectMapper();

        Configuration.ConfigurationDTO configuration = mapper.readValue(new File("./src/main/resources/appConfig.json"), Configuration.ConfigurationDTO.class);

        FillWithArguments(configuration);
        return configuration;
    }

    public void FillWithArguments(ConfigurationDTO configuration) {
        if (this.dataSetId != null) {
            System.out.println("dataSetId NOT NULL: " + this.dataSetId);
            configuration.dataSetId = this.dataSetId;
        } else {
            System.out.println("dataSetId NOT NULL");
        }
        if (this.id != null) {
            configuration.id = this.id;
        }
        if (this.port != null) {
            configuration.port = this.port;
        } else {
            System.out.println("PORT IS NULL");
        }

    }

    public static class ConfigurationDTO {
        public Boolean useCuda;
        public int RAMInGB;
        public Messages.InstanceType instanceType;

        public String datapath;
        public String testdatapath;
        public String datafilename;
        public String targetfilename;
        public int epochs;
        public String id;
        public String host;
        public int port;
        public String address;
        public String pathToModules;
        public String pathToModulesList;
        public String pathToChannel;
        public int dataSetId;
    }
}


