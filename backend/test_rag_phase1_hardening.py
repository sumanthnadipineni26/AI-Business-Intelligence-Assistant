import pytest
from fastapi.testclient import TestClient
from main import app
from app.db.session import SessionLocal
from app.db.models import User, Dataset, DataRecord, DatasetColumn
from app.core.security import get_password_hash

client = TestClient(app)

def get_auth_headers(email: str, password: str, full_name: str, company: str = "Acme Corp", role: str = "Analyst"):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if not user:
            user = User(
                email=email,
                hashed_password=get_password_hash(password),
                full_name=full_name,
                company_name=company,
                role=role
            )
            db.add(user)
            db.commit()
            db.refresh(user)
    finally:
        db.close()

    res = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    assert res.status_code == 200, f"Login failed for {email}: {res.text}"
    token = res.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture(scope="module")
def setup_hardening_test_data():
    """
    Sets up isolated test users and test datasets for Phase 1.1 hardening verification.
    """
    db = SessionLocal()
    created_dataset_ids = []
    created_user_ids = []

    try:
        # User 1: Acme Corp Analyst 1
        u1 = db.query(User).filter(User.email == "analyst_hard1@acme.com").first()
        if not u1:
            u1 = User(
                email="analyst_hard1@acme.com",
                hashed_password=get_password_hash("SecretPass1!"),
                full_name="Acme Analyst One",
                company_name="Acme Corp",
                role="Analyst"
            )
            db.add(u1)
            db.commit()
            db.refresh(u1)
        created_user_ids.append(u1.id)

        # User 2: Acme Corp Analyst 2 (Colleague in same company)
        u2 = db.query(User).filter(User.email == "analyst_hard2@acme.com").first()
        if not u2:
            u2 = User(
                email="analyst_hard2@acme.com",
                hashed_password=get_password_hash("SecretPass1!"),
                full_name="Acme Analyst Two",
                company_name="Acme Corp",
                role="Analyst"
            )
            db.add(u2)
            db.commit()
            db.refresh(u2)
        created_user_ids.append(u2.id)

        # User 3: Rival Corp Analyst (different company)
        u3 = db.query(User).filter(User.email == "analyst_hard@rival.com").first()
        if not u3:
            u3 = User(
                email="analyst_hard@rival.com",
                hashed_password=get_password_hash("SecretPass1!"),
                full_name="Rival Analyst",
                company_name="Rival Corp",
                role="Analyst"
            )
            db.add(u3)
            db.commit()
            db.refresh(u3)
        created_user_ids.append(u3.id)

        # User 4: Admin User
        u4 = db.query(User).filter(User.email == "admin_hard@acme.com").first()
        if not u4:
            u4 = User(
                email="admin_hard@acme.com",
                hashed_password=get_password_hash("SecretPass1!"),
                full_name="Acme Admin",
                company_name="Acme Corp",
                role="Admin"
            )
            db.add(u4)
            db.commit()
            db.refresh(u4)
        created_user_ids.append(u4.id)

        # Helper to clean up any pre-existing datasets
        for ds_id in [201, 202, 203, 204]:
            old_ds = db.query(Dataset).filter(Dataset.id == ds_id).first()
            if old_ds:
                db.query(DataRecord).filter(DataRecord.dataset_id == ds_id).delete()
                db.query(DatasetColumn).filter(DatasetColumn.dataset_id == ds_id).delete()
                db.delete(old_ds)
                db.commit()

        # Dataset 201: Owned by Acme Analyst 1, Year 2025 data only
        ds201 = Dataset(
            id=201,
            owner_id=u1.id,
            company_name="Acme Corp",
            owner_name="Acme Analyst One",
            name="dataset_2025.csv",
            file_type="csv",
            row_count=2,
            column_count=5,
            quality_score=100.0,
            status="processed"
        )
        db.add(ds201)
        db.commit()
        created_dataset_ids.append(201)

        cols201 = [
            DatasetColumn(dataset_id=201, column_name="date", data_type="date"),
            DatasetColumn(dataset_id=201, column_name="region", data_type="string"),
            DatasetColumn(dataset_id=201, column_name="category", data_type="string"),
            DatasetColumn(dataset_id=201, column_name="revenue", data_type="float"),
            DatasetColumn(dataset_id=201, column_name="profit", data_type="float"),
        ]
        db.add_all(cols201)
        db.commit()

        recs201 = [
            {"date": "2025-05-15", "region": "North", "category": "Software", "revenue": 20000.0, "profit": 5000.0},
            {"date": "2025-10-20", "region": "South", "category": "Hardware", "revenue": 15000.0, "profit": 3500.0},
        ]
        for idx, row in enumerate(recs201):
            db.add(DataRecord(dataset_id=201, row_index=idx, payload=row))
        db.commit()

        # Dataset 202: Unowned dataset with NO company (owner_id=None, company_name="")
        ds202 = Dataset(
            id=202,
            owner_id=None,
            company_name="",
            owner_name=None,
            name="unowned_no_company.csv",
            file_type="csv",
            row_count=1,
            column_count=3,
            quality_score=100.0,
            status="processed"
        )
        db.add(ds202)
        db.commit()
        created_dataset_ids.append(202)

        cols202 = [
            DatasetColumn(dataset_id=202, column_name="date", data_type="date"),
            DatasetColumn(dataset_id=202, column_name="revenue", data_type="float"),
            DatasetColumn(dataset_id=202, column_name="profit", data_type="float"),
        ]
        db.add_all(cols202)
        db.commit()
        db.add(DataRecord(dataset_id=202, row_index=0, payload={"date": "2025-01-01", "revenue": 1000.0, "profit": 200.0}))
        db.commit()

        # Dataset 203: Unowned dataset belonging to Acme Corp (owner_id=None, company_name="Acme Corp")
        ds203 = Dataset(
            id=203,
            owner_id=None,
            company_name="Acme Corp",
            owner_name="System Ingest",
            name="unowned_acme_company.csv",
            file_type="csv",
            row_count=1,
            column_count=3,
            quality_score=100.0,
            status="processed"
        )
        db.add(ds203)
        db.commit()
        created_dataset_ids.append(203)

        cols203 = [
            DatasetColumn(dataset_id=203, column_name="date", data_type="date"),
            DatasetColumn(dataset_id=203, column_name="revenue", data_type="float"),
            DatasetColumn(dataset_id=203, column_name="profit", data_type="float"),
        ]
        db.add_all(cols203)
        db.commit()
        db.add(DataRecord(dataset_id=203, row_index=0, payload={"date": "2026-02-01", "revenue": 5000.0, "profit": 1000.0}))
        db.commit()

        # Dataset 204: Multi-year dataset owned by Acme Analyst 1 (dates in 2025 and 2026)
        ds204 = Dataset(
            id=204,
            owner_id=u1.id,
            company_name="Acme Corp",
            owner_name="Acme Analyst One",
            name="multi_year_dataset.csv",
            file_type="csv",
            row_count=3,
            column_count=4,
            quality_score=100.0,
            status="processed"
        )
        db.add(ds204)
        db.commit()
        created_dataset_ids.append(204)

        cols204 = [
            DatasetColumn(dataset_id=204, column_name="date", data_type="date"),
            DatasetColumn(dataset_id=204, column_name="region", data_type="string"),
            DatasetColumn(dataset_id=204, column_name="revenue", data_type="float"),
            DatasetColumn(dataset_id=204, column_name="profit", data_type="float"),
        ]
        db.add_all(cols204)
        db.commit()

        recs204 = [
            {"date": "2025-05-10", "region": "North", "revenue": 50000.0, "profit": 12000.0},
            {"date": "2026-01-15", "region": "South", "revenue": 30000.0, "profit": 7000.0},
            {"date": "2026-05-12", "region": "North", "revenue": 70000.0, "profit": 18000.0},
        ]
        for idx, row in enumerate(recs204):
            db.add(DataRecord(dataset_id=204, row_index=idx, payload=row))
        db.commit()

        yield {
            "u1_email": "analyst_hard1@acme.com",
            "u2_email": "analyst_hard2@acme.com",
            "rival_email": "analyst_hard@rival.com",
            "admin_email": "admin_hard@acme.com",
            "ds201_id": 201,
            "ds202_id": 202,
            "ds203_id": 203,
            "ds204_id": 204
        }

    finally:
        for ds_id in created_dataset_ids:
            ds = db.query(Dataset).filter(Dataset.id == ds_id).first()
            if ds:
                db.query(DataRecord).filter(DataRecord.dataset_id == ds_id).delete()
                db.query(DatasetColumn).filter(DatasetColumn.dataset_id == ds_id).delete()
                db.delete(ds)
        db.commit()
        db.close()


# =====================================================================
# Task 2 Tests: Unowned Dataset & Access Control Matrix
# =====================================================================

def test_access_own_dataset(setup_hardening_test_data):
    """User can access their own dataset."""
    headers = get_auth_headers("analyst_hard1@acme.com", "SecretPass1!", "Acme Analyst One")
    ds_id = setup_hardening_test_data["ds201_id"]
    res = client.get(f"/api/v1/analytics/dashboard?dataset_id={ds_id}", headers=headers)
    assert res.status_code == 200


def test_access_same_company_shared_dataset(setup_hardening_test_data):
    """Colleague in same company can access dataset under legitimate company-sharing rule."""
    headers = get_auth_headers("analyst_hard2@acme.com", "SecretPass1!", "Acme Analyst Two")
    ds_id = setup_hardening_test_data["ds201_id"]
    res = client.get(f"/api/v1/analytics/dashboard?dataset_id={ds_id}", headers=headers)
    assert res.status_code == 200


def test_access_cross_tenant_denied(setup_hardening_test_data):
    """User from another company is denied access with HTTP 403."""
    rival_headers = get_auth_headers("analyst_hard@rival.com", "SecretPass1!", "Rival Analyst", company="Rival Corp")
    ds_id = setup_hardening_test_data["ds201_id"]
    res = client.get(f"/api/v1/analytics/dashboard?dataset_id={ds_id}", headers=rival_headers)
    assert res.status_code == 403
    assert "Access to specified dataset denied" in res.json()["detail"]


def test_unowned_dataset_denied_without_matching_company(setup_hardening_test_data):
    """An unowned dataset with no company is denied to non-admins."""
    headers = get_auth_headers("analyst_hard1@acme.com", "SecretPass1!", "Acme Analyst One")
    ds202_id = setup_hardening_test_data["ds202_id"]
    res = client.get(f"/api/v1/analytics/dashboard?dataset_id={ds202_id}", headers=headers)
    assert res.status_code == 403
    assert "Access to specified dataset denied" in res.json()["detail"]


def test_unowned_dataset_denied_to_cross_tenant(setup_hardening_test_data):
    """An unowned dataset belonging to Acme Corp is denied to Rival Corp user."""
    rival_headers = get_auth_headers("analyst_hard@rival.com", "SecretPass1!", "Rival Analyst", company="Rival Corp")
    ds203_id = setup_hardening_test_data["ds203_id"]
    res = client.get(f"/api/v1/analytics/dashboard?dataset_id={ds203_id}", headers=rival_headers)
    assert res.status_code == 403
    assert "Access to specified dataset denied" in res.json()["detail"]


def test_unowned_dataset_allowed_to_same_company(setup_hardening_test_data):
    """An unowned dataset belonging to Acme Corp is accessible by Acme Corp users."""
    headers = get_auth_headers("analyst_hard1@acme.com", "SecretPass1!", "Acme Analyst One")
    ds203_id = setup_hardening_test_data["ds203_id"]
    res = client.get(f"/api/v1/analytics/dashboard?dataset_id={ds203_id}", headers=headers)
    assert res.status_code == 200


def test_admin_access_unowned_and_cross_tenant(setup_hardening_test_data):
    """Administrator has full access across owned, unowned, and all datasets."""
    admin_headers = get_auth_headers("admin_hard@acme.com", "SecretPass1!", "Acme Admin", role="Admin")
    for ds_key in ["ds201_id", "ds202_id", "ds203_id", "ds204_id"]:
        ds_id = setup_hardening_test_data[ds_key]
        res = client.get(f"/api/v1/analytics/dashboard?dataset_id={ds_id}", headers=admin_headers)
        assert res.status_code == 200


# =====================================================================
# Task 1 Tests: Analytics Endpoints Authorization Enforcement
# =====================================================================

def test_analytics_endpoints_enforce_dataset_authorization(setup_hardening_test_data):
    """
    Confirm all 6 analytics endpoints reject cross-tenant access with HTTP 403:
    1. /insights/{id}
    2. /anomalies/{id}
    3. /dashboard
    4. /kpis
    5. /charts
    6. /health
    """
    rival_headers = get_auth_headers("analyst_hard@rival.com", "SecretPass1!", "Rival Analyst", company="Rival Corp")
    owner_headers = get_auth_headers("analyst_hard1@acme.com", "SecretPass1!", "Acme Analyst One")
    ds_id = setup_hardening_test_data["ds201_id"]

    endpoints = [
        ("GET", f"/api/v1/analytics/insights/{ds_id}"),
        ("GET", f"/api/v1/analytics/anomalies/{ds_id}"),
        ("GET", f"/api/v1/analytics/dashboard?dataset_id={ds_id}"),
        ("GET", f"/api/v1/analytics/kpis?dataset_id={ds_id}"),
        ("GET", f"/api/v1/analytics/charts?dataset_id={ds_id}"),
        ("GET", f"/api/v1/analytics/health?dataset_id={ds_id}"),
    ]

    for method, url in endpoints:
        # Cross-tenant attempt must fail with 403
        res_denied = client.get(url, headers=rival_headers)
        assert res_denied.status_code == 403, f"Endpoint {url} failed to deny unauthorized user: {res_denied.status_code}"
        assert "Access to specified dataset denied" in res_denied.json()["detail"]

        # Authorized owner attempt must succeed with 200
        res_allowed = client.get(url, headers=owner_headers)
        assert res_allowed.status_code == 200, f"Endpoint {url} rejected authorized user: {res_allowed.status_code}"


# =====================================================================
# Task 3 Tests: Dataset-Aware Date Filtering & Ambiguity Handling
# =====================================================================

def test_date_filter_month_in_2025(setup_hardening_test_data):
    """Verify month filtering for 2025 on a 2025 dataset."""
    headers = get_auth_headers("analyst_hard1@acme.com", "SecretPass1!", "Acme Analyst One")
    ds_id = setup_hardening_test_data["ds201_id"]

    # Dataset 201 has row with 2025-05-15 (revenue 20000.0)
    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": ds_id}, headers=headers)
    conv_id = res_conv.json()["id"]

    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What was the total revenue in May 2025?", "dataset_id": ds_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "20,000.00" in ans or "20000" in ans
    assert "35,000" not in ans  # Full dataset sum is 35,000


def test_date_filter_quarter_in_2026(setup_hardening_test_data):
    """Verify quarter filtering for Q1 2026 on a dataset with 2026 records."""
    headers = get_auth_headers("analyst_hard1@acme.com", "SecretPass1!", "Acme Analyst One")
    ds_id = setup_hardening_test_data["ds204_id"]

    # Dataset 204: 2026-01-15 (rev 30,000), 2025-05-10 (rev 50,000), 2026-05-12 (rev 70,000)
    # Q1 2026 sum should be 30,000
    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": ds_id}, headers=headers)
    conv_id = res_conv.json()["id"]

    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What was the total revenue in Q1 2026?", "dataset_id": ds_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "30,000.00" in ans or "30000" in ans


def test_date_filter_dataset_containing_multiple_years(setup_hardening_test_data):
    """Verify explicit year queries resolve correctly on a multi-year dataset."""
    headers = get_auth_headers("analyst_hard1@acme.com", "SecretPass1!", "Acme Analyst One")
    ds_id = setup_hardening_test_data["ds204_id"]

    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": ds_id}, headers=headers)
    conv_id = res_conv.json()["id"]

    # Query May 2025: should be 50,000
    res2025 = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What was the total revenue in May 2025?", "dataset_id": ds_id},
        headers=headers
    )
    assert res2025.status_code == 200
    assert "50,000.00" in res2025.json()["message_text"]

    # Query May 2026: should be 70,000
    res2026 = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What was the total revenue in May 2026?", "dataset_id": ds_id},
        headers=headers
    )
    assert res2026.status_code == 200
    assert "70,000.00" in res2026.json()["message_text"]


def test_date_filter_no_matching_records(setup_hardening_test_data):
    """Verify period with no matching records reports clear abstention without 500 error."""
    headers = get_auth_headers("analyst_hard1@acme.com", "SecretPass1!", "Acme Analyst One")
    ds_id = setup_hardening_test_data["ds201_id"]

    # Dataset 201 has records in May 2025 and Oct 2025, none in February 2025
    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": ds_id}, headers=headers)
    conv_id = res_conv.json()["id"]

    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What was the total revenue in February 2025?", "dataset_id": ds_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "no records found" in ans.lower()
    assert "February 2025" in ans


def test_date_filter_ambiguous_month_queries_abstains_when_multiple_years_present(setup_hardening_test_data):
    """
    When dataset has multiple years (2025 and 2026) and query asks for 'May' without a year,
    the system must abstain and ask for clarification rather than choosing silently.
    """
    headers = get_auth_headers("analyst_hard1@acme.com", "SecretPass1!", "Acme Analyst One")
    ds_id = setup_hardening_test_data["ds204_id"]

    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": ds_id}, headers=headers)
    conv_id = res_conv.json()["id"]

    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What was the total revenue in May?", "dataset_id": ds_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "without specifying a year" in ans or "multiple years" in ans
    assert "2025" in ans and "2026" in ans
