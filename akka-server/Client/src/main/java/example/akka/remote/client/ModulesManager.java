package example.akka.remote.client;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.ObjectCodec;
import com.fasterxml.jackson.databind.DeserializationContext;
import com.fasterxml.jackson.databind.JsonDeserializer;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.module.SimpleModule;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class ModulesManager {
    public String taskId;
    public String fileName;

    // Returns saved modules. Reads data from json file
    public static List<ModuleDTO> GetAvailableModules() {
        String path = null;
        try {
            Configuration configurationHandler = new Configuration();
            Configuration.ConfigurationDTO configuration = configurationHandler.get();

            path = configuration.pathToModulesList;

            File f = new File(path);
            if (!f.exists() || f.isDirectory()) {
                return new ArrayList();
            }
            ObjectMapper mapper = GetMapper();

            List<ModuleDTO> modules = mapper.readValue(f, mapper.getTypeFactory().constructCollectionType(List.class, ModuleDTO.class));
            return modules;
        } catch (IOException e) {
            e.printStackTrace();
        }
        return new ArrayList();
    }

    private static ObjectMapper GetMapper() {
        ObjectMapper mapper = new ObjectMapper();
        SimpleModule simpleModule = new SimpleModule();
        simpleModule.addDeserializer(ModuleDTO.class, new ModuleDeserializer());
        mapper.registerModule(simpleModule);
        return mapper;
    }

    // Saves module received from the server
    public static void SaveModule(String taskId, String fileName) {
        ModuleDTO newModule = new ModuleDTO(taskId, fileName);

        List<ModuleDTO> modules = GetAvailableModules();

        modules.add(newModule);

        try {
            Configuration configurationHandler = new Configuration();
            Configuration.ConfigurationDTO configuration = configurationHandler.get();

            String path = configuration.pathToModulesList;

            File f = new File(path);
            if (!f.exists() || f.isDirectory()) {
                f.createNewFile();
            }
            ObjectMapper mapper = GetMapper();

            String json = mapper.writeValueAsString(modules);
            Path p = Paths.get(path);
            Files.write(p, json.getBytes());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static class ModuleDTO {
        public ModuleDTO(String taskId, String fileName) {
            this.taskId = taskId;
            this.fileName = fileName;
        }
        public String taskId;
        public String fileName;
    }

    // Modules deserializer
    public static class ModuleDeserializer extends JsonDeserializer {
        @Override
        public ModuleDTO deserialize(JsonParser jsonParser,
                                        DeserializationContext deserializationContext) throws IOException {
            ObjectCodec oc = jsonParser.getCodec();
            JsonNode node = oc.readTree(jsonParser);

            // TODO make it more readable
            String taskId = node.get("taskId").asText();
            String fileName = node.get("fileName").asText();

            System.out.println("taskId -> " + taskId);
            ModuleDTO clientModule =  new ModuleDTO(taskId, fileName);
            return clientModule;
        }
    }
}
