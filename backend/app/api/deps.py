from typing import List, Callable
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from jose import JWTError, jwt

from app.core.config import settings
from app.db.session import get_db
from app.db.models import User, Dataset

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception
        user_id = int(user_id_str)
    except (JWTError, ValueError):
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return user

def require_roles(allowed_roles: List[str]) -> Callable:
    def role_checker(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles and current_user.role != "Admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation not permitted. Required roles: {allowed_roles}, your role: {current_user.role}"
            )
        return current_user
    return role_checker

def check_dataset_access(db: Session, user: User, dataset_id: int) -> Dataset:
    """
    Validates that a dataset exists and the user is authorized to access it:
    - Admin users have full access across all datasets.
    - Owners have full access to their uploaded datasets.
    - Users in the same company/tenant have access to organization-shared datasets.
    - Unowned datasets are denied by default unless authorized by Admin or matching company.
    Raises 404 if dataset not found, and 403 if access is denied.
    """
    dataset = db.query(Dataset).filter(Dataset.id == dataset_id).first()
    if not dataset:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found."
        )

    # 1. Admin bypass
    if user.role == "Admin":
        return dataset

    # 2. Personal ownership
    if dataset.owner_id is not None and dataset.owner_id == user.id:
        return dataset

    # 3. Legitimate shared access within same company
    if (
        dataset.company_name
        and user.company_name
        and dataset.company_name.strip().lower() == user.company_name.strip().lower()
    ):
        return dataset

    # 4. Denied
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Access to specified dataset denied."
    )

