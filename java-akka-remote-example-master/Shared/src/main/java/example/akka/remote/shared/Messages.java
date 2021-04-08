package example.akka.remote.shared;

import akka.actor.ActorRef;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.List;

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
        public int port;

        public JoinRoundRequest(LocalDateTime availabilityEndAt, String taskId, String clientId, int port) {
            this.availabilityEndAt = availabilityEndAt;
            this.taskId = taskId;
            this.clientId = clientId;
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

    public static class InformAggregatorAboutNewParticipant implements Serializable {
        public ActorRef deviceReference;
        public int port;
        public String clientId;
        public InformAggregatorAboutNewParticipant(ActorRef deviceReference, String clientId, int port) {
            this.deviceReference = deviceReference;
            this.port = port;
            this.clientId = clientId;
        }
    }

    public static class StartLearningProcessCommand implements Serializable {
        public StartLearningProcessCommand() { }
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

    public enum InstanceType {
        Computer,
        Phone,
    }
}
