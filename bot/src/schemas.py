from pydantic import BaseModel, UUID4, StrictStr
from typing import Optional


class GetMessageRequestModel(BaseModel):
    """
    Входная модель (POST /get_message):
    - dialog_id: UUID текущего диалога
    - last_msg_text: последнее сообщение пользователя
    - last_message_id: (опционально) ID этого сообщения
    """
    dialog_id: UUID4
    last_msg_text: StrictStr
    last_message_id: Optional[UUID4] = None


class GetMessageResponseModel(BaseModel):
    """
    Ответная модель (POST /get_message):
    - new_msg_text: текст, сгенерированный ботом
    - dialog_id: UUID диалога
    """
    new_msg_text: StrictStr
    dialog_id: UUID4
