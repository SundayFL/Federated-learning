package example.akka.remote.shared;

import akka.actor.ActorRef;

import java.io.Serializable;
import java.time.LocalDateTime;

public class Messages {

    public static class JoinRoundRequest implements Serializable {
        public LocalDateTime availabilityEndAt;
        public int learningTaskId;
        public int clientId;

        public JoinRoundRequest(LocalDateTime availabilityEndAt, int learningTaskId, int clientId) {
            this.availabilityEndAt = availabilityEndAt;
            this.learningTaskId = learningTaskId;
            this.clientId = clientId;
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
        public InformAggregatorAboutNewParticipant(ActorRef deviceReference) {
            this.deviceReference = deviceReference;
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
}
