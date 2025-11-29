from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel
from typing import Optional, Any, Dict

from ..config import get_settings
from .agent_api import AgentApiClient


# Pydantic models for request validation
class AuthLoginRequest(BaseModel):
    login: str
    password: str


class BookingCreateRequest(BaseModel):
    patient_id: int
    service_id: int
    date: str
    time: str


class RescheduleRequest(BaseModel):
    booking_id: int
    new_date: str
    new_time: str


class CustomerCreateRequest(BaseModel):
    name: str
    phone: str
    gender: Optional[str] = None
    date_of_birth: Optional[str] = None


class CustomerUpdateRequest(BaseModel):
    name: Optional[str] = None
    phone: Optional[str] = None
    gender: Optional[str] = None
    date_of_birth: Optional[str] = None


class RatingCreateRequest(BaseModel):
    booking_id: int
    rating: int
    comment: Optional[str] = None


class MemoUpsertRequest(BaseModel):
    key: str
    value: Any


router = APIRouter()


def client() -> AgentApiClient:
    return AgentApiClient()


def upstream_headers(request: Request, token: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    ua = request.headers.get("user-agent")
    if ua:
        headers["User-Agent"] = ua
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@router.post("/auth/login")
async def auth_login(body: AuthLoginRequest, request: Request, api: AgentApiClient = Depends(client)):
    async def _call():
        async with api._client() as http:
            resp = await http.post(f"{api.base_url}/auth/login", json=body.dict(), headers=upstream_headers(request))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.post("/auth/refresh")
async def auth_refresh(request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.post(f"{api.base_url}/auth/refresh", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


# Agent Memo endpoints (simplified - using session manager)
@router.post("/agent/memo")
async def memo_upsert(body: MemoUpsertRequest):
    """Store temporary data in session"""
    from ..memory.session_manager import SessionManager
    
    session_manager = SessionManager()
    session_manager.put(body.key, body.value, ttl_minutes=45)
    
    return {"ok": True, "key": body.key}


@router.get("/agent/memo")
async def memo_get(key: str):
    """Retrieve temporary data from session"""
    from ..memory.session_manager import SessionManager
    
    session_manager = SessionManager()
    value = session_manager.get(key)
    
    if value is None:
        raise HTTPException(status_code=404, detail="not_found: no memo found")
    
    return {"key": key, "value": value}


@router.delete("/agent/memo")
async def memo_delete(key: str):
    """Delete temporary data from session"""
    from ..memory.session_manager import SessionManager
    
    session_manager = SessionManager()
    session_manager.delete(key)
    
    return {"ok": True}


@router.post("/auth/logout")
async def auth_logout(request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.post(f"{api.base_url}/auth/logout", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return {"ok": True}
    return await _call()


@router.get("/booking")
async def booking_list(
    request: Request,
    api: AgentApiClient = Depends(client),
    patient_id: int | None = None,
    doctor_id: int | None = None,
    state: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
    order: str | None = None,
):
    token = await api.get_jwt()
    params: dict[str, str] = {}
    for k, v in {
        "patient_id": patient_id,
        "doctor_id": doctor_id,
        "state": state,
        "date_from": date_from,
        "date_to": date_to,
        "limit": limit,
        "offset": offset,
        "order": order,
    }.items():
        if v is not None:
            params[k] = str(v)
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/booking", params=params, headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/booking/{rid}")
async def booking_get(rid: str, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/booking/{rid}", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.post("/booking/create")
async def booking_create(body: BookingCreateRequest, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.post(f"{api.base_url}/booking/create", json=body.dict(), headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.post("/sessions/reschedule")
async def sessions_reschedule(body: RescheduleRequest, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.post(f"{api.base_url}/sessions/reschedule", json=body.dict(), headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/patients")
async def patients_list(request: Request, api: AgentApiClient = Depends(client), q: str | None = None, limit: int | None = None, offset: int | None = None, order: str | None = None):
    token = await api.get_jwt()
    params: dict[str, str] = {}
    for k, v in {"q": q, "limit": limit, "offset": offset, "order": order}.items():
        if v is not None:
            params[k] = str(v)
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/patients", params=params, headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/patients/{rid}")
async def patients_get(rid: str, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/patients/{rid}", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.post("/customer/create")
async def customer_create(body: CustomerCreateRequest, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.post(f"{api.base_url}/customer/create", json=body.dict(), headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.put("/customer/update/{rid}")
async def customer_update(rid: str, body: CustomerUpdateRequest, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.put(f"{api.base_url}/customer/update/{rid}", json=body.dict(), headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/services")
async def services_list(request: Request, api: AgentApiClient = Depends(client), service_type_id: int | None = None, q: str | None = None, limit: int | None = None, offset: int | None = None, order: str | None = None):
    token = await api.get_jwt()
    params: dict[str, str] = {}
    for k, v in {"service_type_id": service_type_id, "q": q, "limit": limit, "offset": offset, "order": order}.items():
        if v is not None:
            params[k] = str(v)
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/services", params=params, headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/services/{rid}")
async def services_get(rid: str, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/services/{rid}", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/doctors")
async def doctors_list(request: Request, api: AgentApiClient = Depends(client), q: str | None = None, limit: int | None = None, offset: int | None = None, order: str | None = None):
    token = await api.get_jwt()
    params: dict[str, str] = {}
    for k, v in {"q": q, "limit": limit, "offset": offset, "order": order}.items():
        if v is not None:
            params[k] = str(v)
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/doctors", params=params, headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/doctors/{rid}")
async def doctors_get(rid: str, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/doctors/{rid}", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/specialists")
async def specialists_list(request: Request, api: AgentApiClient = Depends(client), q: str | None = None, limit: int | None = None, offset: int | None = None, order: str | None = None):
    token = await api.get_jwt()
    params: dict[str, str] = {}
    for k, v in {"q": q, "limit": limit, "offset": offset, "order": order}.items():
        if v is not None:
            params[k] = str(v)
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/specialists", params=params, headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/specialists/{rid}")
async def specialists_get(rid: str, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/specialists/{rid}", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/devices")
async def devices_list(request: Request, api: AgentApiClient = Depends(client), q: str | None = None, limit: int | None = None, offset: int | None = None, order: str | None = None):
    token = await api.get_jwt()
    params: dict[str, str] = {}
    for k, v in {"q": q, "limit": limit, "offset": offset, "order": order}.items():
        if v is not None:
            params[k] = str(v)
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/devices", params=params, headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/devices/{rid}")
async def devices_get(rid: str, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/devices/{rid}", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/slots")
async def slots_list(request: Request, api: AgentApiClient = Depends(client), service_id: int | None = None, date: str | None = None, device_id: int | None = None, doctor_id: int | None = None, specialist_id: int | None = None, duration_minutes: int | None = None):
    token = await api.get_jwt()
    params: dict[str, str] = {}
    for k, v in {"service_id": service_id, "date": date, "device_id": device_id, "doctor_id": doctor_id, "specialist_id": specialist_id, "duration_minutes": duration_minutes}.items():
        if v is not None:
            params[k] = str(v)
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/slots", params=params, headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/rating")
async def rating_list(request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/rating", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/rating/{rid}")
async def rating_get(rid: str, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/rating/{rid}", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.post("/rating/create")
async def rating_create(body: RatingCreateRequest, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.post(f"{api.base_url}/rating/create", json=body.dict(), headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.post("/wallet/create")
async def wallet_create(body: dict, request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.post(f"{api.base_url}/wallet/create", json=body, headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


@router.get("/wallet")
async def wallet_get(request: Request, api: AgentApiClient = Depends(client)):
    token = await api.get_jwt()
    async def _call():
        async with api._client() as http:
            resp = await http.get(f"{api.base_url}/wallet", headers=upstream_headers(request, token))
            resp.raise_for_status()
            return resp.json()
    return await _call()


