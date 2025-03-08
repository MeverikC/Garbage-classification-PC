from flask_login import UserMixin
from datetime import datetime

# 模拟用户数据
USERS = {
    "admin": {"password": "admin123", "role": "admin", "created_at": "2018-02-02", "updated_at": "2025-02-18"},
    "user": {"password": "user123", "role": "user", "created_at": "2018-02-02", "updated_at": "2025-02-23"},
}


class User(UserMixin):
    def __init__(self, username, password, role):
        self.id = username
        self.password = password
        self.role = role

    @classmethod
    def get(cls, username):
        user_data = USERS.get(username)
        if user_data:
            return cls(username, user_data["password"], user_data["role"])
        return None

    # 新增用户（静态方法）
    @staticmethod
    def add_user(username, password, role):
        if username in USERS:
            return False  # 用户已存在

        USERS[username] = {
            "password": password,
            "role": role,
            "created_at": datetime.now().strftime("%Y-%m-%d"),
            "updated_at": datetime.now().strftime("%Y-%m-%d")
        }
        return True

    # 删除用户（静态方法）
    @staticmethod
    def delete_user(username):
        if username in USERS:
            del USERS[username]
            return True
        return False  # 用户不存在

    # 编辑用户信息（静态方法）
    @staticmethod
    def edit_user(self, edit_username, username, password=None, role=None):
        if edit_username not in USERS:
            return False  # 用户不存在

        if username != edit_username:
            edit_created_at = USERS[edit_username]["created_at"]
            self.delete_user(edit_username)
            if password and role:
                USERS[username] = {
                    "password": password,
                    "role": role,
                    "created_at": edit_created_at,
                    "updated_at": datetime.now().strftime("%Y-%m-%d")
                }
                return True
            return False

        # 只更新传入的字段
        if password:
            USERS[edit_username]["password"] = password
        if role:
            USERS[edit_username]["role"] = role

        USERS[edit_username]["updated_at"] = datetime.now().strftime("%Y-%m-%d")
        return True
