import jwt
import datetime
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from config import Config

_bearer = HTTPBearer(auto_error=False)

# Marks tokens minted by the password+TOTP login. Pre-P3 builds handed out 365-day
# tokens from an unauthenticated GET /token, signed with the same JWT_SECRET;
# requiring this claim (and a lifetime within JWT_TTL_HOURS) makes those worthless.
_SCOPE = "owner-2fa"

def generate_token() -> str:
    payload = {
        "user": "prithvi",
        "scope": _SCOPE,
        "iat": datetime.datetime.utcnow(),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=Config.JWT_TTL_HOURS)
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
        claims = jwt.decode(token, Config.JWT_SECRET, algorithms=["HS256"],
                            options={"require": ["exp", "iat", "scope"]})
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    if (claims.get("scope") != _SCOPE
            or claims["exp"] - claims["iat"] > Config.JWT_TTL_HOURS * 3600):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token"
        )
    return claims
