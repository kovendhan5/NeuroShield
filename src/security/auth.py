"""JWT Authentication for NeuroShield API.

Provides token generation, validation, and dependency injection for FastAPI.
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthCredential

# Configuration
SECRET_KEY = os.getenv("API_SECRET_KEY", "change-me-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRY_HOURS = 24

security = HTTPBearer()


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a JWT access token.

    Args:
        subject: Unique identifier (usually user/service ID)
        expires_delta: Custom expiry time

    Returns:
        JWT token string
    """
    if expires_delta is None:
        expires_delta = timedelta(hours=TOKEN_EXPIRY_HOURS)

    to_encode = {
        "sub": subject,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + expires_delta,
    }

    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> str:
    """Verify JWT token and extract subject.

    Args:
        token: JWT token string

    Returns:
        Subject claim from token

    Raises:
        HTTPException: If token invalid or expired
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        subject: str = payload.get("sub")
        if subject is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing subject"
            )
        return subject
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token: {e}"
        )


async def get_current_user(
    credentials: HTTPAuthCredential = Depends(security)
) -> str:
    """FastAPI dependency to validate JWT from Authorization header.

    Usage:
        @app.get("/protected")
        async def protected_route(user: str = Depends(get_current_user)):
            return {"user": user}

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        Subject (user/service identifier)

    Raises:
        HTTPException: If token invalid or missing
    """
    token = credentials.credentials
    return verify_token(token)


async def get_current_user_optional(
    credentials: HTTPAuthCredential = Depends(security)
) -> Optional[str]:
    """Optional JWT validation - returns None if token invalid.

    Useful for endpoints that work with or without authentication.
    """
    try:
        token = credentials.credentials
        return verify_token(token)
    except HTTPException:
        return None
