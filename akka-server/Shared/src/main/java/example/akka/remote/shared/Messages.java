package example.akka.remote.shared;

import akka.actor.ActorRef;

import java.io.Serializable;
import java.security.PublicKey;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

public class Messages {

    public static class StartLearning implements Serializable {
        public String id;

        public StartLearning(String id) {
            this.id = id;
        }
    }

    public static class JoinRoundRequest implements Serializable {
        public LocalDateTime availabilityEndAt;
        public String taskId;
        public String clientId;
        public String address;
        public int port;

        public JoinRoundRequest(LocalDateTime availabilityEndAt, String taskId, String clientId, String address, int port) {
            this.availabilityEndAt = availabilityEndAt;
            this.taskId = taskId;
            this.clientId = clientId;
            this.address = address;
            this.port = port;
        }
    }

    public static class JoinRoundResponse implements Serializable {
        public boolean isLearningAvailable;
        public ActorRef aggregator;

        public JoinRoundResponse(boolean isLearningAvailable, ActorRef aggregator) {
            this.isLearningAvailable = isLearningAvailable;
            this.aggregator = aggregator;
        }
    }

    public static class Sum implements Serializable {
        private int first;
        private int second;

        public Sum(int first, int second) {
            this.first = first;
            this.second = second;
        }

        public int getFirst() {
            return first;
        }

        public int getSecond() {
            return second;
        }
    }

    public static class Result implements Serializable {
        private int result;

        public Result(int result) {
            this.result = result;
        }

        public int getResult() {
            return result;
        }
    }

    // Run module message
    public static class RunModule implements Serializable {
        public String moduleFileName;
        public String modelConfig;
        public RunModule(String moduleFileName, String modelConfig) {
            this.moduleFileName = moduleFileName;
            this.modelConfig = modelConfig;
        }
    }

    public static class InformAggregatorAboutNewParticipant implements Serializable {
        public ActorRef deviceReference;
        public int port;
        public String clientId;
        public String address;
        public InformAggregatorAboutNewParticipant(ActorRef deviceReference, String clientId, String address, int port) {
            this.deviceReference = deviceReference;
            this.port = port;
            this.clientId = clientId;
            this.address = address;
        }
    }

    public static class StartLearningProcessCommand implements Serializable {
        public String modelConfig;
        public boolean secureAggr;
        public double DP_threshold;

        public StartLearningProcessCommand(String modelConfig, boolean secureAggr, double DP_threshold) {
            this.modelConfig = modelConfig;
            this.secureAggr = secureAggr;
            this.DP_threshold = DP_threshold;
        }

        public String getModelConfig() {
            return modelConfig;
        }


        public double getDP_threshold() {
            return DP_threshold;
        }
    }

    public static class StartRoundCoordinatorSelector implements Serializable {
        public ActorRef aggregator;

        public StartRoundCoordinatorSelector(ActorRef aggregator) {
            this.aggregator = aggregator;
        }
    }

    public static class StartLearningModule implements Serializable {

        public StartLearningModule() {

        }
    }

    public static class GetModulesListRequest implements Serializable {
        public String id;

        public GetModulesListRequest(String id) {
            this.id = id;
        }
    }

    public static class GetModulesListResponse implements Serializable {
        public List<ModuleData> modules;

        public GetModulesListResponse(List<ModuleData> modules) {
            this.modules = modules;
        }
    }

    public static class GetModuleRequest implements Serializable {
        public String name;

        public GetModuleRequest(String name) {
            this.name = name;
        }
    }

    public static class GetModuleResponse implements Serializable {
        public byte[] content;
        public String fileName;

        public GetModuleResponse(byte[] content, String fileName) {
            this.content = content;
            this.fileName = fileName;
        }
    }

    public static class ModuleData implements Serializable {
        public ModuleData(String id, String fileName, String description, Boolean useCUDA, int minRAMInGB, InstanceType instanceType) {
            this.id = id;
            this.fileName = fileName;
            this.description = description;
            this.useCUDA = useCUDA;
            this.minRAMInGB = minRAMInGB;
            this.instanceType = instanceType;
        }

        public String id;
        public String fileName;
        public String description;

        public Boolean useCUDA;
        public int minRAMInGB;
        public InstanceType instanceType;
    }

    public static class ContactData implements Serializable {
        public ContactData(ActorRef reference, Float publicKey){
            this.reference=reference;
            this.publicKey=publicKey;
        }
        public ActorRef reference;
        public Float publicKey;
    }

    public static class ClientDataSpread implements Serializable {
        public ClientDataSpread(String clientId,
                                int numberOfClients,
                                Map<String, PublicKey> publicKeys,
                                boolean secureAgg,
                                boolean diffPriv,
                                double DP_threshold,
                                double DP_std){
            this.clientId = clientId;
            this.numberOfClients = numberOfClients;
            this.publicKeys = publicKeys;
            this.secureAgg = secureAgg;
            this.diffPriv = diffPriv;
            this.DP_threshold = DP_threshold;
            this.DP_std = DP_std;
        }

        public String clientId;
        public int numberOfClients;
        public Map<String, PublicKey> publicKeys;
        public boolean secureAgg;
        public boolean diffPriv;
        public double DP_threshold;
        public double DP_std;
    }

    public static class AreYouAliveQuestion implements Serializable { }

    public static class IAmAlive implements Serializable {
        public IAmAlive(PublicKey publicKey){
            this.publicKey = publicKey;
        }
        public PublicKey publicKey;
    }

    public static class StartRound implements Serializable { }

    public static class RValuesReady implements Serializable { }

    public static class SendRValue implements Serializable {
        public String sender;
        public byte[] bytes;
        public byte[] key;
        public String receiver;

        public SendRValue(String sender, byte[] bytes, byte[] key, String receiver){
            this.sender=sender;
            this.bytes=bytes;
            this.key=key;
            this.receiver=receiver;
        }
    }

    public static class SendInterRes implements Serializable {
        public String sender;
        public byte[] bytes;

        public SendInterRes(String sender, byte[] bytes){
            this.sender=sender;
            this.bytes=bytes;
        }
    }

    public static class TestMyModel implements Serializable {
        public byte[] bytes;

        public TestMyModel(byte[] bytes) {
            this.bytes = bytes;
        }
    }

    public static class TestResults implements Serializable {
        public byte[] bytes;
        public String id;

        public TestResults(byte[] bytes, String id) {
            this.bytes = bytes;
            this.id = id;
        }
    }

    public static class RoundEnded implements Serializable { }

    public enum InstanceType {
        Computer,
        Phone,
    }
}
