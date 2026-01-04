from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app import crud, models, schemas
from app.api import deps

router = APIRouter()

@router.get("/conversations", response_model=List[schemas.Conversation])
def read_conversations(
    *,
    db: Session = Depends(deps.get_db),
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取当前用户的所有会话
    """
    # 获取所有会话元组列表 (user_id, latest_message, unread_count)
    conversations_data = crud.message.get_conversations(db=db, user_id=current_user.id)
    
    # 转换为Conversation模型
    result = []
    for user_id, latest_message, unread_count in conversations_data:
        # 获取用户信息
        other_user = crud.user.get(db=db, id=user_id)
        if not other_user:
            continue
            
        conversation = schemas.Conversation(
            id=user_id,  # 使用对话用户ID作为会话ID
            participants=[current_user.id, user_id],
            last_message=latest_message,
            unread_count=unread_count
        )
        result.append(conversation)
    
    return result

@router.get("/{user_id}", response_model=List[schemas.Message])
def read_messages(
    *,
    db: Session = Depends(deps.get_db),
    user_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    获取与特定用户的消息历史
    """
    messages = crud.message.get_conversation(
        db=db, user_id1=current_user.id, user_id2=user_id
    )
    return messages

@router.post("/", response_model=schemas.Message)
def send_message(
    *,
    db: Session = Depends(deps.get_db),
    message_in: schemas.MessageCreate,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    发送私信
    """
    # 检查接收者是否存在
    receiver = crud.user.get(db=db, id=message_in.receiver_id)
    if not receiver:
        raise HTTPException(status_code=404, detail="Receiver not found")
    
    message = crud.message.create_with_sender(
        db=db, obj_in=message_in, sender_id=current_user.id
    )
    return message

@router.post("/{message_id}/read", response_model=schemas.Message)
def mark_message_read(
    *,
    db: Session = Depends(deps.get_db),
    message_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    标记消息为已读
    """
    message = crud.message.get(db, id=message_id)
    if not message:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="消息不存在",
        )
    
    # 检查消息接收者是否为当前用户
    if message.receiver_id != current_user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="没有足够权限",
        )
    
    # 标记为已读
    message = crud.message.mark_one_as_read(db, message_id=message_id)
    return message

@router.post("/{user_id}/read-all", response_model=int)
def mark_all_as_read(
    *,
    db: Session = Depends(deps.get_db),
    user_id: int,
    current_user: models.User = Depends(deps.get_current_active_user),
) -> Any:
    """
    标记与特定用户的所有消息为已读
    """
    # 将用户发送给当前用户的所有消息标记为已读
    count = crud.message.mark_as_read(
        db=db, user_id1=current_user.id, user_id2=user_id
    )
    return count 