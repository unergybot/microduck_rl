"""V2 navigation facade over the single existing durable motion owner."""

import math
import uuid
from threading import Lock

from .contracts import TaskCommandRequest, TaskCreateRequest
from .navigation_contracts import (
    Binding,
    Environment,
    NavigationTaskRequest,
    Pose,
    digest,
)
from .service import (
    InvalidParameters,
    NotReady,
    StaleCommand,
    TaskConflict,
    TaskNotFound,
)


class NavigationRuntimeRequest(TaskCreateRequest):
    """Private start material, deliberately not accepted by the public V1 model."""

    navigation: NavigationTaskRequest


class NavigationTaskService:
    def __init__(self, service, installation, *, enabled=False):
        self.service = service
        self.installation = installation
        self.enabled = enabled
        self.session = uuid.uuid4().hex
        self.lock = Lock()
        self.sequence = 0

    def capabilities(self):
        if not self.enabled or self.installation is None:
            return {"ready": False, "reasonCodes": ["NAVIGATION_DISABLED"]}
        if not any(
            a.actionCode == "WALK_VELOCITY" and a.availability == "AVAILABLE"
            for a in self.service._bundle.actions
        ):
            return {"ready": False, "reasonCodes": ["LOCOMOTION_UNAVAILABLE"]}
        ready, reasons = self.service.motion_readiness()
        status = self.service.observe_navigation()
        installation = self.installation
        generation = self.service.runtime_generation()
        with self.lock:
            self.sequence += 1
            sequence = self.sequence
        q = status.baseOrientationXyzw
        yaw = math.atan2(
            2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        )
        environment = Environment(
            schema="rom.microduck.environment.v1",
            binding=Binding(
                robotId=installation.robot_id,
                targetId=installation.target_id,
                runtimeSession=f"{self.session}:{generation}",
            ),
            provenance="SIM_GROUND_TRUTH",
            capturedAt=status.timestamp.isoformat().replace("+00:00", "Z"),
            sequence=sequence,
            mapFrame="map",
            bodyFrame="body",
            transformRevision="xy-forward-left-z-up-v1",
            pose=Pose(x=status.basePositionM[0], y=status.basePositionM[1], yaw=yaw),
            valid=not status.fallen and not status.limp,
            scene=installation.scene,
            mapDigest=digest(installation.scene),
            bundleDigest=installation.bundle_digest,
            navigationProfileDigest=digest(installation.profile),
        )
        from .navigation.grid import Grid
        from .navigation.planner import plan_cells

        grid = Grid(installation.scene, installation.profile)
        reachable = [
            name
            for name, goal in installation.scene.landmarks.items()
            if grid.free(environment.pose.x, environment.pose.y)
            and grid.free(goal.x, goal.y)
            and plan_cells(
                grid.blocked,
                grid.cell(environment.pose.x, environment.pose.y),
                grid.cell(goal.x, goal.y),
                grid.bounds,
            )
            is not None
        ]
        return {
            "reachableLandmarks": reachable,
            "ready": ready and environment.valid,
            "reasonCodes": list(reasons),
            "environment": environment.model_dump(mode="json", by_alias=True),
            "profile": installation.profile.model_dump(mode="json"),
            "bundleVersion": self.service._bundle.bundleVersion,
            "bundleId": self.service._bundle.bundleId,
            "qualificationDigest": installation.qualification_digest,
            "evaluationStatus": installation.evaluation_status,
        }

    def _request(self, task_id):
        raw = self.service._store.request_content(task_id)
        if raw is None or "navigation" not in raw:
            raise TaskNotFound("navigation task not found")
        return NavigationTaskRequest.model_validate(raw["navigation"])

    def _snapshot(self, task_id):
        request = self._request(task_id)
        snapshot = self.service.get_task(task_id).model_dump(mode="json", by_alias=True)
        snapshot.update(
            schema="MICRODUCK_NAVIGATION_STATE_V2",
            actionCode="NAVIGATE_TO_LANDMARK",
            proposalDigest=request.proposalDigest,
            binding=request.proposal.binding.model_dump(),
            mapDigest=request.proposal.mapDigest,
            navigationProfileDigest=request.proposal.navigationProfileDigest,
            provenance="SIM_GROUND_TRUTH",
            landmarkId=request.proposal.landmarkId,
        )
        return snapshot

    def create_task(self, request):
        # Recover an uncertain submit even if readiness/session changed in the meantime.
        existing = self.service._store.request_content(request.taskId)
        if existing is not None:
            if (
                "navigation" not in existing
                or NavigationTaskRequest.model_validate(existing["navigation"])
                != request
            ):
                raise TaskConflict("navigation task identity conflict")
            return self._snapshot(request.taskId)
        caps = self.capabilities()
        if not caps["ready"]:
            raise NotReady("navigation is unavailable")
        env = caps["environment"]
        prop = request.proposal
        if (
            prop.binding.model_dump() != env["binding"]
            or prop.mapDigest != env["mapDigest"]
            or prop.bundleDigest != env["bundleDigest"]
            or prop.navigationProfileDigest != env["navigationProfileDigest"]
            or prop.bundleVersion != caps["bundleVersion"]
        ):
            raise InvalidParameters("navigation proposal is stale")
        from .navigation.grid import Grid
        from .navigation.planner import plan_cells

        grid = Grid(prop.scene, prop.profile)
        pose = env["pose"]
        if (
            plan_cells(
                grid.blocked,
                grid.cell(pose["x"], pose["y"]),
                grid.cell(prop.destination.x, prop.destination.y),
                grid.bounds,
            )
            is None
        ):
            raise InvalidParameters("navigation destination unreachable")
        internal = NavigationRuntimeRequest(
            schema="MICRODUCK_SIM_TASK_V1",
            taskId=request.taskId,
            actionCode="WALK_VELOCITY",
            bundleVersion=prop.bundleVersion,
            bundleDigest=prop.bundleDigest,
            parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
            scenario={"terrain": "flat", "seed": 7},
            leaseMs=prop.profile.leaseMs,
            requestedBy=request.requestedBy,
            navigation=request,
        )
        self.service.create_task(internal)
        return self._snapshot(request.taskId)

    def get_task(self, task_id):
        return self._snapshot(task_id)

    def renew_lease(self, task_id, renewal):
        with self.lock:
            request = self._request(task_id)
            if not self.enabled:
                self.service.cancel_task(task_id)
                raise NotReady("navigation is disabled")
            if (
                renewal.taskId != task_id
                or renewal.proposalDigest != request.proposalDigest
            ):
                raise InvalidParameters("navigation authorization identity mismatch")
            previous = self.service._store.command_sequence(task_id)
            if previous is not None and renewal.sequence <= previous:
                raise StaleCommand("navigation sequence must increase")
            self.service.command(
                task_id,
                TaskCommandRequest(
                    commandSequence=renewal.sequence,
                    parameters={"vxMps": 0.0, "vyMps": 0.0, "yawRateRadps": 0.0},
                    leaseMs=request.proposal.profile.leaseMs,
                ),
                navigation=True,
            )
            return self._snapshot(task_id)

    def cancel_task(self, task_id):
        self._request(task_id)
        self.service.cancel_task(task_id)
        return self._snapshot(task_id)

    def events_after(self, task_id, sequence, page_size=100):
        self._request(task_id)
        return {
            "events": [
                e.model_dump(mode="json", by_alias=True)
                for e in self.service.events_after(
                    task_id, sequence, page_size=page_size
                )
            ]
        }
