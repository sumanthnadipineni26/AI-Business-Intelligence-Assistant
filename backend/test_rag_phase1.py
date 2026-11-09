import pytest
from fastapi.testclient import TestClient
from main import app
from app.db.session import SessionLocal
from app.db.models import User, Dataset, DataRecord, DatasetColumn, Conversation, ChatMessageModel
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
def setup_phase1_test_data():
    """
    Sets up isolated test users and test datasets for Phase 1 verification.
    """
    db = SessionLocal()
    created_dataset_ids = []
    created_user_ids = []

    try:
        # User 1: Acme Corp Analyst
        u1 = db.query(User).filter(User.email == "analyst_p1@acme.com").first()
        if not u1:
            u1 = User(
                email="analyst_p1@acme.com",
                hashed_password=get_password_hash("SecretPass1!"),
                full_name="Acme Analyst",
                company_name="Acme Corp",
                role="Analyst"
            )
            db.add(u1)
            db.commit()
            db.refresh(u1)
        created_user_ids.append(u1.id)

        # User 2: Rival Corp Analyst (different tenant)
        u2 = db.query(User).filter(User.email == "analyst_p1@rival.com").first()
        if not u2:
            u2 = User(
                email="analyst_p1@rival.com",
                hashed_password=get_password_hash("SecretPass1!"),
                full_name="Rival Analyst",
                company_name="Rival Corp",
                role="Analyst"
            )
            db.add(u2)
            db.commit()
            db.refresh(u2)
        created_user_ids.append(u2.id)

        # Dataset 101: Filtered Sales Dataset owned by Acme Analyst
        ds1 = db.query(Dataset).filter(Dataset.id == 101).first()
        if ds1:
            db.query(DataRecord).filter(DataRecord.dataset_id == 101).delete()
            db.query(DatasetColumn).filter(DatasetColumn.dataset_id == 101).delete()
            db.delete(ds1)
            db.commit()

        ds1 = Dataset(
            id=101,
            owner_id=u1.id,
            company_name="Acme Corp",
            owner_name="Acme Analyst",
            name="phase1_test_sales.csv",
            file_type="csv",
            row_count=5,
            column_count=8,
            quality_score=100.0,
            status="processed"
        )
        db.add(ds1)
        db.commit()
        created_dataset_ids.append(101)

        cols1 = [
            DatasetColumn(dataset_id=101, column_name="date", data_type="date"),
            DatasetColumn(dataset_id=101, column_name="region", data_type="string"),
            DatasetColumn(dataset_id=101, column_name="product", data_type="string"),
            DatasetColumn(dataset_id=101, column_name="category", data_type="string"),
            DatasetColumn(dataset_id=101, column_name="customer", data_type="string"),
            DatasetColumn(dataset_id=101, column_name="revenue", data_type="float"),
            DatasetColumn(dataset_id=101, column_name="profit", data_type="float"),
            DatasetColumn(dataset_id=101, column_name="orders", data_type="integer"),
        ]
        db.add_all(cols1)
        db.commit()

        records_data1 = [
            {"date": "2026-01-05", "region": "North", "product": "Laptop", "category": "Electronics", "customer": "ABC Corp", "revenue": 12000.0, "profit": 2500.0, "orders": 15},
            {"date": "2026-01-12", "region": "South", "product": "Phone", "category": "Electronics", "customer": "XYZ Ltd", "revenue": 8000.0, "profit": 1800.0, "orders": 22},
            {"date": "2026-02-03", "region": "East", "product": "Chair", "category": "Furniture", "customer": "HomePlus", "revenue": 3500.0, "profit": 700.0, "orders": 18},
            {"date": "2026-03-01", "region": "North", "product": "Monitor", "category": "Electronics", "customer": "ABC Corp", "revenue": 6200.0, "profit": 1400.0, "orders": 16},
            {"date": "2026-05-11", "region": "North", "product": "Laptop", "category": "Electronics", "customer": "Global Inc", "revenue": 15000.0, "profit": 3200.0, "orders": 19},
        ]
        for idx, row in enumerate(records_data1):
            db.add(DataRecord(dataset_id=101, row_index=idx, payload=row))
        db.commit()

        # Dataset 102: Empty Dataset (0 rows)
        ds2 = db.query(Dataset).filter(Dataset.id == 102).first()
        if ds2:
            db.query(DataRecord).filter(DataRecord.dataset_id == 102).delete()
            db.query(DatasetColumn).filter(DatasetColumn.dataset_id == 102).delete()
            db.delete(ds2)
            db.commit()

        ds2 = Dataset(
            id=102,
            owner_id=u1.id,
            company_name="Acme Corp",
            owner_name="Acme Analyst",
            name="empty_dataset.csv",
            file_type="csv",
            row_count=0,
            column_count=3,
            quality_score=0.0,
            status="processed"
        )
        db.add(ds2)
        db.commit()
        created_dataset_ids.append(102)

        # Dataset 103: All-Null Column Dataset
        ds3 = db.query(Dataset).filter(Dataset.id == 103).first()
        if ds3:
            db.query(DataRecord).filter(DataRecord.dataset_id == 103).delete()
            db.query(DatasetColumn).filter(DatasetColumn.dataset_id == 103).delete()
            db.delete(ds3)
            db.commit()

        ds3 = Dataset(
            id=103,
            owner_id=u1.id,
            company_name="Acme Corp",
            owner_name="Acme Analyst",
            name="null_columns_dataset.csv",
            file_type="csv",
            row_count=3,
            column_count=3,
            quality_score=50.0,
            status="processed"
        )
        db.add(ds3)
        db.commit()
        created_dataset_ids.append(103)

        records_data3 = [
            {"item_id": 1, "all_null_metric": None, "all_null_category": None},
            {"item_id": 2, "all_null_metric": None, "all_null_category": None},
            {"item_id": 3, "all_null_metric": None, "all_null_category": None},
        ]
        for idx, row in enumerate(records_data3):
            db.add(DataRecord(dataset_id=103, row_index=idx, payload=row))
        db.commit()

        yield {
            "sales_dataset_id": 101,
            "empty_dataset_id": 102,
            "null_dataset_id": 103,
            "acme_user": "analyst_p1@acme.com",
            "rival_user": "analyst_p1@rival.com"
        }

    finally:
        # Cleanup created records
        for did in created_dataset_ids:
            try:
                db.query(DataRecord).filter(DataRecord.dataset_id == did).delete()
                db.query(DatasetColumn).filter(DatasetColumn.dataset_id == did).delete()
                db.query(Conversation).filter(Conversation.dataset_id == did).delete()
                ds = db.query(Dataset).filter(Dataset.id == did).first()
                if ds:
                    db.delete(ds)
                db.commit()
            except Exception:
                db.rollback()
        db.close()


def test_chat_missing_dataset_id_returns_400(setup_phase1_test_data):
    """Confirm that omitting dataset_id when none is established returns 400 Bad Request."""
    headers = get_auth_headers("analyst_p1@acme.com", "SecretPass1!", "Acme Analyst")
    
    # 1. Create conversation with no dataset_id
    res_conv = client.post("/api/v1/chat/conversations", json={"title": "Unbound Conv", "dataset_id": None}, headers=headers)
    assert res_conv.status_code == 200
    conv_id = res_conv.json()["id"]

    # 2. Post message with no dataset_id in chat body
    res_msg = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is our total revenue?", "dataset_id": None},
        headers=headers
    )
    assert res_msg.status_code == 400
    assert "Active dataset_id is required" in res_msg.json()["detail"]

    # 3. Direct analytics chat endpoint also requires dataset_id
    res_direct = client.post(
        "/api/v1/analytics/chat",
        json={"message": "What is our total revenue?", "dataset_id": None},
        headers=headers
    )
    assert res_direct.status_code == 400
    assert "Active dataset_id is required" in res_direct.json()["detail"]


def test_chat_nonexistent_dataset_id_returns_404(setup_phase1_test_data):
    """Confirm that passing a non-existent dataset_id returns 404 Not Found."""
    headers = get_auth_headers("analyst_p1@acme.com", "SecretPass1!", "Acme Analyst")
    
    # Attempt to create conversation with non-existent dataset
    res = client.post("/api/v1/chat/conversations", json={"dataset_id": 999999}, headers=headers)
    assert res.status_code == 404
    assert "Dataset not found" in res.json()["detail"]

    # Attempt to query message with non-existent dataset
    res_conv = client.post("/api/v1/chat/conversations", json={"title": "Valid Conv"}, headers=headers)
    conv_id = res_conv.json()["id"]
    res_msg = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is the revenue?", "dataset_id": 999999},
        headers=headers
    )
    assert res_msg.status_code == 404
    assert "Dataset not found" in res_msg.json()["detail"]


def test_chat_unauthorized_dataset_access_returns_403(setup_phase1_test_data):
    """Confirm cross-tenant access is rejected with 403 Forbidden."""
    rival_headers = get_auth_headers("analyst_p1@rival.com", "SecretPass1!", "Rival Analyst", company="Rival Corp")
    sales_id = setup_phase1_test_data["sales_dataset_id"]

    # Rival user attempting to create conversation using Acme's dataset #101
    res = client.post("/api/v1/chat/conversations", json={"dataset_id": sales_id}, headers=rival_headers)
    assert res.status_code == 403
    assert "Access to specified dataset denied" in res.json()["detail"]


def test_chat_prevents_silent_fallback(setup_phase1_test_data):
    """Verify system never falls back to another user's or tenant's dataset silently."""
    rival_headers = get_auth_headers("analyst_p1@rival.com", "SecretPass1!", "Rival Analyst", company="Rival Corp")
    
    # Create conversation with no dataset specified
    res_conv = client.post("/api/v1/chat/conversations", json={"title": "Rival Conv", "dataset_id": None}, headers=rival_headers)
    assert res_conv.status_code == 200
    conv_id = res_conv.json()["id"]
    
    # Send message: must not silently attach Acme's dataset #101
    res_msg = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is total revenue?", "dataset_id": None},
        headers=rival_headers
    )
    assert res_msg.status_code == 400
    assert "Active dataset_id is required" in res_msg.json()["detail"]


def test_chat_filtered_revenue_and_profit_calculations(setup_phase1_test_data):
    """
    Verify exact filtered calculations:
    Dataset rows:
      0: North, Laptop, Electronics, ABC Corp, rev=12000, prof=2500, orders=15
      1: South, Phone, Electronics, XYZ Ltd, rev=8000, prof=1800, orders=22
      2: East, Chair, Furniture, HomePlus, rev=3500, prof=700, orders=18
      3: North, Monitor, Electronics, ABC Corp, rev=6200, prof=1400, orders=16
      4: North, Laptop, Electronics, Global Inc, rev=15000, prof=3200, orders=19
    Global sum = 44,700 revenue, 9,600 profit.
    North sum = 12000 + 6200 + 15000 = 33,200.00.
    """
    headers = get_auth_headers("analyst_p1@acme.com", "SecretPass1!", "Acme Analyst")
    sales_id = setup_phase1_test_data["sales_dataset_id"]

    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": sales_id}, headers=headers)
    conv_id = res_conv.json()["id"]

    # 1. Filtered North region revenue (must return 33,200, not 44,700)
    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is the total revenue in the North region?", "dataset_id": sales_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "33,200.00" in ans or "33200" in ans
    assert "44,700" not in ans

    # 2. Filtered Electronics category revenue (12000 + 8000 + 6200 + 15000 = 41,200)
    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is the total revenue for the Electronics category?", "dataset_id": sales_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "41,200.00" in ans or "41200" in ans

    # 3. Filtered customer revenue (ABC Corp = 12000 + 6200 = 18,200)
    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is the total revenue from customer ABC Corp?", "dataset_id": sales_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "18,200.00" in ans or "18200" in ans

    # 4. Filtered profit in North region (2500 + 1400 + 3200 = 7,100.00)
    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is the total profit generated in the North region?", "dataset_id": sales_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "7,100.00" in ans or "7100" in ans


def test_chat_product_comparison(setup_phase1_test_data):
    """
    Phone orders = 22.
    Laptop orders = 15 (row 0) + 19 (row 4) = 34.
    Laptop (34) > Phone (22).
    """
    headers = get_auth_headers("analyst_p1@acme.com", "SecretPass1!", "Acme Analyst")
    sales_id = setup_phase1_test_data["sales_dataset_id"]

    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": sales_id}, headers=headers)
    conv_id = res_conv.json()["id"]

    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "Between Phone and Laptop, which product had more total orders placed?", "dataset_id": sales_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "Laptop" in ans
    assert "34" in ans
    assert "22" in ans


def test_chat_empty_dataset_handling(setup_phase1_test_data):
    """Confirm empty dataset produces clear informative message without raising ValueError/HTTP 500."""
    headers = get_auth_headers("analyst_p1@acme.com", "SecretPass1!", "Acme Analyst")
    empty_id = setup_phase1_test_data["empty_dataset_id"]

    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": empty_id}, headers=headers)
    assert res_conv.status_code == 200
    conv_id = res_conv.json()["id"]

    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is the total revenue?", "dataset_id": empty_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "empty" in ans.lower() or "no data" in ans.lower() or "0 records" in ans.lower()


def test_chat_all_null_column_safety(setup_phase1_test_data):
    """Confirm all-null column does not raise IndexError or ValueError."""
    headers = get_auth_headers("analyst_p1@acme.com", "SecretPass1!", "Acme Analyst")
    null_id = setup_phase1_test_data["null_dataset_id"]

    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": null_id}, headers=headers)
    assert res_conv.status_code == 200
    conv_id = res_conv.json()["id"]

    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is the average of all_null_metric?", "dataset_id": null_id},
        headers=headers
    )
    assert res.status_code == 200
    ans = res.json()["message_text"]
    assert "undefined" in ans.lower() or "no valid" in ans.lower() or "no non-null" in ans.lower()


def test_chat_unsupported_columns_and_abstention(setup_phase1_test_data):
    """Confirm clear abstention for unsupported columns, non-existent regions, and missing domains."""
    headers = get_auth_headers("analyst_p1@acme.com", "SecretPass1!", "Acme Analyst")
    sales_id = setup_phase1_test_data["sales_dataset_id"]

    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": sales_id}, headers=headers)
    conv_id = res_conv.json()["id"]

    # 1. Non-existent shipping cost
    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What was the shipping cost for this order?", "dataset_id": sales_id},
        headers=headers
    )
    assert res.status_code == 200
    assert "shipping" in res.json()["message_text"].lower()
    assert "not contain" in res.json()["message_text"].lower()

    # 2. Non-existent region EMEA
    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is the total revenue for the EMEA region?", "dataset_id": sales_id},
        headers=headers
    )
    assert res.status_code == 200
    assert "EMEA" in res.json()["message_text"]
    assert "not contain" in res.json()["message_text"].lower()

    # 3. Student GPA on sales dataset
    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is the average student GPA?", "dataset_id": sales_id},
        headers=headers
    )
    assert res.status_code == 200
    assert "student marks/gpa" in res.json()["message_text"].lower()


def test_chat_response_schema_compatibility(setup_phase1_test_data):
    """Verify response complies strictly with ChatMessageResponse schema and UI expectations."""
    headers = get_auth_headers("analyst_p1@acme.com", "SecretPass1!", "Acme Analyst")
    sales_id = setup_phase1_test_data["sales_dataset_id"]

    res_conv = client.post("/api/v1/chat/conversations", json={"dataset_id": sales_id}, headers=headers)
    conv_id = res_conv.json()["id"]

    res = client.post(
        f"/api/v1/chat/conversations/{conv_id}/messages",
        json={"message": "What is the total revenue?", "dataset_id": sales_id},
        headers=headers
    )
    assert res.status_code == 200
    data = res.json()
    assert "id" in data
    assert "conversation_id" in data
    assert "sender" in data and data["sender"] == "assistant"
    assert "message_text" in data
    assert "sources_json" in data
    assert isinstance(data["sources_json"], list)
    assert "confidence_score" in data
    assert "created_at" in data
