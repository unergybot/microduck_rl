"""Private navigation routes, sharing the simulator's existing authentication."""

from fastapi import Depends, Path, Query

from .navigation_contracts import NavigationLeaseRequest, NavigationTaskRequest
from .service import NotReady


def install_routes(app, service, require_bearer):
    def navigation():
        result = getattr(service, "navigation", None)
        if result is None:
            raise NotReady("navigation is unavailable")
        return result

    dependencies = [Depends(require_bearer)]

    @app.get("/v2/navigation/capabilities", dependencies=dependencies)
    def capabilities():
        nav = getattr(service, "navigation", None)
        return (
            nav.capabilities()
            if nav
            else {"ready": False, "reasonCodes": ["NAVIGATION_DISABLED"]}
        )

    @app.post("/v2/navigation/tasks", dependencies=dependencies, status_code=202)
    def create(request: NavigationTaskRequest):
        return navigation().create_task(request)

    @app.get("/v2/navigation/tasks/{task_id}", dependencies=dependencies)
    def task(task_id: str = Path(pattern=r"^[0-9a-f]{32}$")):
        return navigation().get_task(task_id)

    @app.put("/v2/navigation/tasks/{task_id}/lease", dependencies=dependencies)
    def lease(
        request: NavigationLeaseRequest, task_id: str = Path(pattern=r"^[0-9a-f]{32}$")
    ):
        return navigation().renew_lease(task_id, request)

    @app.post("/v2/navigation/tasks/{task_id}/cancel", dependencies=dependencies)
    def cancel(task_id: str = Path(pattern=r"^[0-9a-f]{32}$")):
        return navigation().cancel_task(task_id)

    @app.get("/v2/navigation/tasks/{task_id}/events", dependencies=dependencies)
    def events(
        task_id: str = Path(pattern=r"^[0-9a-f]{32}$"),
        afterSequence: int = Query(default=-1, ge=-1),
        pageSize: int = Query(default=100, ge=1, le=100),
    ):
        return navigation().events_after(task_id, afterSequence, pageSize)
