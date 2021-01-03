package com.example;

import akka.actor.typed.javadsl.*;
import akka.actor.typed.ActorRef;

public class DeviceManager extends AbstractBehavior<DeviceManager.Command> {

    public DeviceManager(ActorContext<Command> context) {
        super(context);
    }

    @Override
    public Receive<Command> createReceive() {
        return null;
    }

    public interface Command {}

    public static final class RequestTrackDevice implements DeviceManager.Command, DeviceGroup.Command {
        public final int requestId;
        public final String groupId;
        public final String deviceId;
        public final ActorRef<DeviceRegistered> replyTo;

        public RequestTrackDevice(int requestId, String groupId, String deviceId, ActorRef<DeviceRegistered> replyTo) {
            this.requestId = requestId;
            this.groupId = groupId;
            this.deviceId = deviceId;
            this.replyTo = replyTo;
        }
    }

    public static final class DeviceRegistered {
        public final int requestId;
        public final ActorRef<Device.Command> device;

        public DeviceRegistered(int requestId, ActorRef<Device.Command> device) {
            this.requestId = requestId;
            this.device = device;
        }
    }
}