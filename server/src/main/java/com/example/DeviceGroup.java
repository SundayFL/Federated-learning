package com.example;

import akka.actor.typed.Behavior;
import akka.actor.typed.javadsl.*;
import akka.actor.typed.ActorRef;

import java.util.HashMap;
import java.util.Map;

public class DeviceGroup extends AbstractBehavior<DeviceGroup.Command> {

    public interface Command {}

    private class DeviceTerminated implements Command {
        public final ActorRef<Device.Command> device;
        public final String groupId;
        public final String deviceId;

        DeviceTerminated(ActorRef<Device.Command> device, String groupId, String deviceId) {
            this.device = device;
            this.groupId = groupId;
            this.deviceId = deviceId;
        }
    }

    public static Behavior<Command> create(String groupId) {
        return Behaviors.setup(context -> new DeviceGroup(context, groupId));
    }

    private final String groupId;
    private final Map<String, ActorRef<Device.Command>> deviceIdToActor = new HashMap<>();

    private DeviceGroup(ActorContext<Command> context, String groupId) {
        super(context);
        this.groupId = groupId;
        context.getLog().info("DeviceGroup {} started", groupId);
    }

    private DeviceGroup onTrackDevice(DeviceManager.RequestTrackDevice trackMsg) {
        if (this.groupId.equals(trackMsg.groupId)) {
            ActorRef<Device.Command> deviceActor = deviceIdToActor.get(trackMsg.deviceId);
            if (deviceActor != null) {
                trackMsg.replyTo.tell(new DeviceManager.DeviceRegistered(trackMsg.requestId, deviceActor));
            } else {
                getContext().getLog().info("Creating device actor for {}", trackMsg.deviceId);
                deviceActor =
                        getContext()
                                .spawn(Device.create(groupId, trackMsg.deviceId), "device-" + trackMsg.deviceId);
                deviceIdToActor.put(trackMsg.deviceId, deviceActor);
                trackMsg.replyTo.tell(new DeviceManager.DeviceRegistered(trackMsg.requestId, deviceActor));
            }
        } else {
            getContext()
                    .getLog()
                    .warn(
                            "Ignoring TrackDevice request for {}. This actor is responsible for {}.",
                            groupId,
                            this.groupId);
        }
        return this;
    }


    @Override
    public Receive<Command> createReceive() {
        return newReceiveBuilder()
                .onMessage(DeviceManager.RequestTrackDevice.class, this::onTrackDevice)
                .build();
    }

    private DeviceGroup onPostStop() {
        getContext().getLog().info("DeviceGroup {} stopped", groupId);
        return this;
    }
}