from fastapi import APIRouter

router = APIRouter()

@router.get("/items/{item_id}")
def read_item(item_id: int):
    return {"item_id": item_id, "description": "This is a sample item."}
