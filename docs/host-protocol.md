# Client API Reference

**English** | [简体中文](zh-CN/host-protocol.md)

`alohamini.client.HostClient` reads state and images and sends named motion targets. Use `alohamini inspect` for read-only CLI queries.

This reference is for custom clients and policy development. For daily teleoperation, collection, and evaluation, see the [user guide](alohamini.md) and [training and deployment](training.md).

## Robot descriptions

```python
from alohamini.model import get_robot_model

model = get_robot_model("alohamini2pro")
physical_xml = model.description_xml("collision")
planning_xml = model.description_xml("planning")
```

`description_xml()` returns ROS XML; `description_tree()` returns an independently editable element tree. Read `planning` and `kinematic` views through these interfaces; use `description_path()` only for actual source files. `description_names` lists available descriptions. Restart affected processes after updating model assets.

## Reading from Python

Replace the address and model with the actual configuration. See the [user guide](alohamini.md) for Raspberry Pi startup.

```python
from alohamini.client import HostClient

with HostClient("robot-host", expected_model="alohamini2pro") as robot:
    snapshot = robot.read()
    print(snapshot.robot_model)
    print(snapshot.payload["_safety"])

    snapshot = robot.read(include_images=True)
    forward_jpeg = snapshot.images.get("forward")
```

## Requests and responses

The client is a DEALER and the Host is a ROUTER, using TCP port 5556 by default.

- State request: a unique token ending in `:state`.
- Request including images: a unique token ending in `:full`.
- Recording image request: `<REQUEST_ID>:<episode UUID>:record`, using a separate bounded camera cursor.
- Response received by the DEALER: `[token, state_json, camera_name, jpeg, ...]`.
- The ROUTER client identity is part of the routing envelope, not the DEALER response contents.
- State requests must return only the token and JSON, with `_images=[]`.
- Camera names and order must exactly match `_images` in the JSON.

At most 3 requests are in flight by default; `request_window` accepts 1–16. A newer matching response supersedes older unanswered requests. Timeout clears in-flight requests while retaining the connection and verified control context; protocol errors close the connection. Neither returns cached state.

Call `set_recording_cameras(True)` at recording start, then use `read(include_images=True)` to receive image groups from the bounded queue. Each enable creates a new episode UUID; recording images are limited to one in-flight request. `_camera_buffer.pending=true` means the group is incomplete. Call `set_recording_cameras(False)` afterward.

## Commands

`send_command(targets, based_on=snapshot)` uses PUSH to Host port 5555. Configure `expected_model` explicitly and call `connect_control()` before the control loop to establish state and action connections. Anonymous commands are not sent to a Host lacking session/epoch information.

- `targets` is a nonempty mapping accepting model arm `*.pos`, `x.vel`, `y.vel`, `theta.vel`, `lift_axis.height_mm`, and `lift_axis.stop`. Values must be finite. The client does not fill omitted axes or expose raw lift-velocity commands.
- On receipt of a ZMQ command, the Host sets omitted base velocity axes to zero; an arm-only command also stops earlier base motion. Omitted arm/lift targets retain their previous targets.
- `lift_axis.stop=1` stops only the lift and is mutually exclusive with a height target. The Host sends zero velocity, waits at least 100 ms, then holds using local height feedback sampled afterward. Without a valid height reference it only stops and does not create a reference. Confirm rest using subsequent feedback.
- Joint positions follow Host calibration and normalization. Base x/y use m/s, theta uses deg/s, and lift targets use mm. See [ROS 2](ros2.md) for ROS unit conversions.
- JSON contains targets and `_command`, whose fields are `client_id`, monotonically increasing `sequence`, `host_session_id`, and `control_epoch`.
- Commands bind to the session and ownership in `based_on`; use a new snapshot after an epoch change. Release by the same client watchdog can recover, but Host restarts or takeover by another client must not resume old actions directly.
- Targets are validated before sending. Unavailable connections do not buffer offline targets, and targets are not resent automatically.
- A returned `CommandIdentity` means the message entered the transport queue, not that the Host accepted it or motors executed it. Later targets may overwrite unsent targets.
- If the channel is temporarily unwritable, returns `None` immediately while retaining the connection. The target was not sent and must not be recorded as a sent action. Generate a target from fresh state on the next cycle instead of resending the old one.

Action submission is nonblocking; state reads wait at most `timeout_s`. Without valid feedback for longer than the Host watchdog period, `send_command()` returns `None`; brief timeouts do not immediately disable sending. Applications manage policy queues; the Host enforces hardware protection and command admission.

The Host watchdog timeout is 2 seconds. Only valid commands that complete hardware writes renew it. Invalid, repeated, or other-owner commands do not renew it; write failures enter fault handling.

`close()` discards unsent messages and closes connections. It does not implicitly send position targets, disable torque, or home the robot. The Host watchdog handles stopping after the controller disconnects.

## State and units

`snapshot.payload` retains response fields, including unknown extensions:

| Field | Meaning |
| --- | --- |
| `_robot_metadata` | `schema_version=1`, model, calibration mapping metadata, and enabled cameras |
| `_host_timing` | Sampling times and clock domains supplied by the Host |
| `_safety` | Host protection, command source, ownership, and session information |
| `_motor_feedback` | Optional motor feedback; fields may be partial |
| `_software` | Optional software identity: `schema_version=1`, installed version, Git commit, and SHA256 of actual Python sources |
| `*.pos` | Existing Host joint position values; units depend on motor normalization |
| `lift_axis.height_mm` | Existing Host lift height in millimeters |
| `lift_axis.homed` | Whether the continuous-encoder height reference is valid; does not imply a height sensor or lower limit switch |
| `lift_axis.raw_tick` | Current single-turn encoder reading; omitted when the reference is invalid |
| `lift_axis.extended_ticks`, `lift_axis.zero_extended_ticks` | Continuous and zero-height counts from the same Host tracker; omitted when the reference is invalid |
| `lift_axis.reference_sequence` | Number of height-reference establishments in this Host session; increments when a new reference is established |

Check version, read times, and valid values before using `_motor_feedback`; do not replace missing fields with zeros.

Lift parameters are in `_robot_metadata.lift_axis`: `ticks_per_revolution`, `lead_mm_per_revolution` (including gearing), and `direction_sign`. Height is `height_mm = direction_sign × (extended_ticks - zero_extended_ticks) × lead_mm_per_revolution / ticks_per_revolution`. Use continuous counts only within the same Host session and valid reference. Sampling tools must check `reference_sequence` and must not join data across reconnections.

`round_trip_s` is the client request round-trip duration. PC and Host monotonic timestamps cannot be subtracted directly; start a new timeline after Host restart.

## Errors and resource boundaries

`ProtocolError` indicates a protocol error; `ModelMismatchError`, a model mismatch; `ResponseTimeoutError`, a timeout; `alohamini.errors.ConnectionError`, a transport error; and `CommandRejectedError`, current client conditions preventing submission. Read errors never return the previous frame.

Duplicate JSON keys, nonfinite numbers, unsupported robot metadata versions, and inconsistent image envelopes are rejected. Limits are 1 MiB JSON, 8 MiB per frame, 32 MiB per response, and 16 cameras. JPEGs are returned as bytes; the client does not decode or validate pixels.

Use each client object in one thread only, and call `close()` or use `with`. Timeouts bound network requests, not real-time scheduling. Connections support IPv4 addresses or hostnames and must be used on trusted networks; the protocol has no authentication or encryption.

## Python policy evaluation

```bash
alohamini evaluate --host <PI_IP> --robot_model alohamini2pro \
  --policy my_policy:load --episode_time 30
```

`my_policy:load` is a no-argument factory in an importable developer module, not a model directory.
The returned object provides:

- `robot_metadata`: model, calibration, and lift parameters recorded with training data; do not substitute configuration read at startup.
- `reset()`: clear policy history, action queues, and preprocessor state; called once per episode.
- `select_action(snapshot)`: accepts a `HostSnapshot` and returns named Python float targets for every arm `*.pos`, three base velocities, and lift height. Uses the Host units above; unnamed vectors and end-effector deltas are not accepted.

The policy handles model loading, image decoding, preprocessing, postprocessing, and output coordinate conversion. See `examples/learning/custom_policy.py`.

Defaults are one episode, 60 seconds, and 30 Hz. Set episode count and reset time with `--num_episodes` and `--reset_time`. `--dataset NAME --task "TASK"` saves inputs, submitted requests, and actual sampling times; inference duration limits the actual rate.

Policies return finite named targets. Using calibration, the Host clamps normally normalized joints to `[-100,100]`, grippers to `[0,100]`, and lift travel to its limits; out-of-range degree joints are rejected. Evaluation records retain the original submitted values.

Evaluation frames store the submitted request in `action` / `requested_action` (including normalized values that may exceed limits), with `action_semantics="host_request_v1"` in the safety sidecar. `policy_action` stores predictions before fixed-dimension overrides. These fields do not confirm execution: the current frame's `_safety` contains feedback received before sending the command. Match `accepted_targets` using the full command identity in subsequent feedback; do not treat it as the outcome of the current request. Training readers do not automatically replace requests with accepted targets.

Evaluation uses cameras already enabled on the Host. It pauses during joint protection or prolonged disconnection, clears the old policy cache on recovery, and resumes. It ends if the Host restarts, another client takes control, calibration changes, or the lift reference changes. On exit, it requests position hold and zero base velocity if it still owns control. The Host watchdog remains active during inference.

## Camera subscription (5557)

When cameras are enabled, the Host provides a separate publication channel on TCP 5557 at the same listening address. Subscribe to `camera/<CAMERA_NAME>` with ZMQ SUB; messages contain `[topic, metadata_json, jpeg]`. `HostClient` does not subscribe to this channel.

- `schema_version=1`; `camera_name` matches the topic, `encoding=jpeg`, and `width/height` describe the rotated image.
- `host_session_id` matches the Host session in the 5556 state; each stream's `sequence` increases within that session. Clients should reset sequence checks when the session changes.
- `capture_monotonic_s` marks completion of the camera read, not exposure time; `capture_unix_ns` is converted using the same machine's `host_clock_reference`. Synchronize system clocks before using wall-clock timestamps across machines.
- JPEG data can be used directly in ROS `CompressedImage`.

Subscriptions use cameras that are already enabled, acquire no motion control, and consume no recording cursor. Slow subscribers may drop frames without blocking Host control. With `--profile_timing`, `[HOST CAMERA STREAM avg ms/frame]` reports subscription-stream encoding time separately.
