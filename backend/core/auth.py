import jwt
import datetime
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from config import Config

_bearer = HTTPBearer(auto_error=False)

def generate_token() -> str:
    payload = {
        "user": "prithvi",
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=365)
    }
    return jwt.encode(payload, Config.JWT_SECRET, algorithm="HS256")

def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
) -> dict:
    """FastAPI dependency: validates the Bearer JWT and returns its payload.

    Apply with `dependencies=[Depends(require_auth)]` on a route or router,
    or inject as a parameter to access the decoded payload.
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="No token"
        )
    token = credentials.credentials
    try:
        return jwt.decode(token, Config.JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
