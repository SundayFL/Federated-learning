package example.akka.remote.server;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;
import example.akka.remote.shared.Messages;

import java.io.File;
import java.io.IOException;
import java.util.List;

public class Configuration {

    // Returns current configuration read from appConfig.json file
    public static ConfigurationDTO get() {
        ObjectMapper mapper = new ObjectMapper();

        SimpleModule simpleModule = new SimpleModule();
        simpleModule.addDeserializer(ClientModule.class, new ClientModuleDeserializer());
        mapper.registerModule(simpleModule);

        ConfigurationDTO configuration = null;
        try {
            configuration = mapper.readValue(new File("./src/main/resources/appConfig.json"), ConfigurationDTO.class);
        } catch (IOException e) {
            e.printStackTrace();
        }
        return configuration;
    }

    public static class ConfigurationDTO {
        public boolean secureAgg;
        public int minimumNumberOfDevices;
        public String learningTaskId;
        public String serverModuleFilePathSA;
        public String serverModuleFilePath;
        public String testDataPath;
        public String pathToResources;
        public String savedModelPathSA;
        public String savedModelPath;
        public int epochs;
        public String modelConfig;
        public int targetOutputSize;
        public double DP_noiseVariance;
        public double DP_threshold;

        @JsonProperty(value = "clientModules")
        public List<ClientModule> clientModules;
    }

    public static class ClientModule {
        public ClientModule(String learningTaskId, String fileName, String description, Boolean useCUDA, int minRAMInGB, Messages.InstanceType instanceType) {
            this.learningTaskId = learningTaskId;
            this.fileName = fileName;
            this.description = description;
            this.useCUDA = useCUDA;
            this.minRAMInGB = minRAMInGB;
            this.instanceType = instanceType;
        }

        public String learningTaskId;
        public String fileName;
        public String description;

        public Boolean useCUDA;
        public int minRAMInGB;
        public Messages.InstanceType instanceType;
    }

    public static class ClientModuleDeserializer extends JsonDeserializer {
        @Override
        public ClientModule deserialize(JsonParser jsonParser,
                                        DeserializationContext deserializationContext) throws IOException {
            ObjectCodec oc = jsonParser.getCodec();
            JsonNode node = oc.readTree(jsonParser);

            // TODO make it more readable
            String learningTaskId = node.get("learningTaskId").asText();
            String fileName = node.get("fileName").asText();
            String description = node.get("description").asText();
            Boolean useCUDA = node.get("useCUDA").asBoolean();
            int minRAMInGB = node.get("minRAMInGB").asInt();
            Messages.InstanceType instanceType = Messages.InstanceType.valueOf(node.get("instanceType").asText());

            System.out.println("learningTaskId -> " + learningTaskId);
            ClientModule clientModule =  new ClientModule(learningTaskId, fileName, description, useCUDA, minRAMInGB, instanceType);
            return clientModule;
        }
    }
}


