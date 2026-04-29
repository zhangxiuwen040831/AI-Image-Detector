from fastapi import APIRouter, HTTPException, Depends, status, Request, Query
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional
import pymysql
from pymysql.cursors import DictCursor
import bcrypt
import jwt
from datetime import datetime, timedelta
import os

# JWT配置
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

# 数据库配置
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'user': os.getenv('DB_USER', 'root'),
    'password': os.getenv('DB_PASSWORD', '123456'),
    'database': os.getenv('DB_NAME', 'ai_image_detector_db'),
    'charset': 'utf8mb4',
    'cursorclass': DictCursor
}

print("DB CONNECT:", DB_CONFIG['host'], DB_CONFIG['user'], DB_CONFIG['database'])

# 创建路由器
router = APIRouter(prefix="/auth", tags=["认证"])
security = HTTPBearer()

# Pydantic模型
class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str
    nickname: Optional[str] = None
    email: Optional[str] = None

class UserResponse(BaseModel):
    id: int
    username: str
    nickname: str
    user_type: str
    last_login_time: Optional[datetime] = None
    last_logout_time: Optional[datetime] = None

class UpdatePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class AdminUpdatePasswordRequest(BaseModel):
    user_id: int
    new_password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class UserListResponse(BaseModel):
    users: list
    total: int

class UserLogsResponse(BaseModel):
    logs: list
    total: int

def get_db_connection():
    """获取数据库连接"""
    return pymysql.connect(**DB_CONFIG)

def ensure_auth_tables():
    """Ensure auth-related tables exist."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                  id INT AUTO_INCREMENT PRIMARY KEY,
                  username VARCHAR(100) NOT NULL UNIQUE,
                  password_hash VARCHAR(255) NOT NULL,
                  nickname VARCHAR(100) NOT NULL DEFAULT '用户',
                  user_type ENUM('normal','admin') NOT NULL DEFAULT 'normal',
                  last_login_time DATETIME NULL,
                  last_logout_time DATETIME NULL,
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
                  is_active TINYINT(1) DEFAULT 1
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_logs (
                  id INT AUTO_INCREMENT PRIMARY KEY,
                  user_id INT NOT NULL,
                  operate_type VARCHAR(50),
                  operate_content VARCHAR(255),
                  operate_ip VARCHAR(50),
                  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                  INDEX idx_user_id (user_id),
                  CONSTRAINT fk_user_logs_user_id FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)
        conn.commit()
    finally:
        conn.close()

def log_user_operation(cursor, user_id: int, operate_type: str, operate_content: str, operate_ip: Optional[str]):
    cursor.execute(
        """
        INSERT INTO user_logs (user_id, operate_type, operate_content, operate_ip)
        VALUES (%s, %s, %s, %s)
        """,
        (user_id, operate_type, operate_content, operate_ip),
    )

def get_request_ip(request: Optional[Request]) -> str:
    if not request:
        return ""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    client = request.client.host if request.client else ""
    return client or ""

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """验证密码"""
    try:
        return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))
    except Exception:
        return False

def hash_password(password: str) -> str:
    """哈希密码"""
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

def create_token(user_id: int, username: str, user_type: str) -> str:
    """创建JWT token"""
    payload = {
        "user_id": user_id,
        "username": username,
        "user_type": user_type,
        "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def decode_token(token: str) -> dict:
    """解码JWT token"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token已过期"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的Token"
        )

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """获取当前登录用户"""
    token = credentials.credentials
    payload = decode_token(token)

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, username, nickname, user_type, last_login_time, last_logout_time FROM users WHERE id = %s AND is_active = TRUE",
                (payload['user_id'],)
            )
            user = cursor.fetchone()
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="用户不存在或已被禁用"
                )
            return user
    finally:
        conn.close()

async def get_current_admin_user(current_user: dict = Depends(get_current_user)):
    """获取当前登录的管理员用户"""
    if current_user['user_type'] != 'admin':
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="需要管理员权限"
        )
    return current_user

@router.post("/register")
async def register(request: RegisterRequest):
    """用户注册"""
    username = (request.username or "").strip()
    password = request.password or ""
    nickname = (request.nickname or "").strip() or "用户"

    if not username or not password:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "message": "请输入用户名和密码"},
        )

    if len(username) < 3:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "message": "用户名长度至少3个字符"},
        )

    if len(password) < 6:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "message": "密码长度至少6个字符"},
        )

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
            if cursor.fetchone():
                return JSONResponse(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    content={"success": False, "message": "用户名已存在"},
                )

            password_hash = hash_password(password)

            cursor.execute("""
                INSERT INTO users (username, password_hash, nickname, user_type, is_active)
                VALUES (%s, %s, %s, 'normal', 1)
            """, (username, password_hash, nickname))

            conn.commit()
            return {
                "success": True,
                "message": "注册成功，请登录",
            }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": f"注册失败: {str(e)}"},
        )
    finally:
        conn.close()

@router.post("/login")
async def login(request: LoginRequest, http_request: Request = None):
    """用户登录"""
    print("LOGIN REQUEST:", request.username)
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "SELECT id, username, password_hash, nickname, user_type, is_active FROM users WHERE username = %s",
                (request.username,)
            )
            user = cursor.fetchone()

            if not user:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"success": False, "message": "用户名或密码错误"},
                )

            if not user['is_active']:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"success": False, "message": "用户已被禁用"},
                )

            password_ok = bcrypt.checkpw(request.password.encode('utf-8'), user['password_hash'].encode('utf-8'))
            print("CHECK:", request.username, password_ok)
            if not password_ok:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"success": False, "message": "用户名或密码错误"},
                )

            cursor.execute(
                "UPDATE users SET last_login_time = NOW() WHERE id = %s",
                (user['id'],)
            )
            log_user_operation(
                cursor,
                user['id'],
                'login',
                '用户登录系统',
                get_request_ip(http_request),
            )
            conn.commit()

            cursor.execute(
                "SELECT id, username, nickname, user_type, last_login_time, last_logout_time FROM users WHERE id = %s",
                (user['id'],)
            )
            user = cursor.fetchone()

            token = create_token(user['id'], user['username'], user['user_type'])

            return {
                "success": True,
                "message": "登录成功",
                "access_token": token,
                "token_type": "bearer",
                "user": {
                    "user_id": user['id'],
                    "username": user['username'],
                    "nickname": user['nickname'],
                    "user_type": user['user_type'],
                    "last_login_time": user['last_login_time'].isoformat() if user.get('last_login_time') else None,
                    "last_logout_time": user['last_logout_time'].isoformat() if user.get('last_logout_time') else None,
                },
            }
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": f"登录失败: {str(e)}"},
        )
    finally:
        conn.close()

@router.post("/logout")
async def logout(current_user: dict = Depends(get_current_user), request: Request = None):
    """用户登出"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET last_logout_time = NOW() WHERE id = %s",
                (current_user['id'],)
            )
            log_user_operation(
                cursor,
                current_user['id'],
                'logout',
                '用户退出系统',
                get_request_ip(request),
            )
            conn.commit()
            return {"success": True, "message": "登出成功"}
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"success": False, "message": f"登出失败: {str(e)}"},
        )
    finally:
        conn.close()

@router.get("/user/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """获取当前用户信息"""
    return UserResponse(**current_user)

@router.put("/user/password")
async def update_password(
    request: UpdatePasswordRequest,
    current_user: dict = Depends(get_current_user),
    http_request: Request = None,
):
    """修改当前用户密码"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 获取当前密码哈希
            cursor.execute(
                "SELECT password_hash FROM users WHERE id = %s",
                (current_user['id'],)
            )
            user = cursor.fetchone()

            # 验证旧密码
            if not verify_password(request.old_password, user['password_hash']):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="原密码错误"
                )

            # 更新新密码
            new_password_hash = hash_password(request.new_password)
            cursor.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (new_password_hash, current_user['id'])
            )
            log_user_operation(
                cursor,
                current_user['id'],
                'update_profile',
                '修改密码',
                get_request_ip(http_request),
            )
            conn.commit()

            return {"success": True, "message": "密码修改成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"密码修改失败: {str(e)}"
        )
    finally:
        conn.close()

@router.put("/user/nickname")
async def update_nickname(
    nickname: str,
    current_user: dict = Depends(get_current_user),
    request: Request = None,
):
    """修改当前用户昵称"""
    if len(nickname) < 1 or len(nickname) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="昵称长度必须在1-100个字符之间"
        )

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                "UPDATE users SET nickname = %s WHERE id = %s",
                (nickname, current_user['id'])
            )
            log_user_operation(
                cursor,
                current_user['id'],
                'update_profile',
                '修改用户信息',
                get_request_ip(request),
            )
            conn.commit()

            # 返回更新后的用户信息
            cursor.execute(
                "SELECT id, username, nickname, user_type, last_login_time, last_logout_time FROM users WHERE id = %s",
                (current_user['id'],)
            )
            user = cursor.fetchone()

            return UserResponse(**user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"昵称修改失败: {str(e)}"
        )
    finally:
        conn.close()

# 管理员接口
@router.get("/admin/users", response_model=UserListResponse)
async def get_user_list(current_admin: dict = Depends(get_current_admin_user)):
    """获取所有普通用户列表（仅管理员）"""
    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                SELECT id, username, nickname, user_type, last_login_time, last_logout_time, created_at, is_active
                FROM users
                ORDER BY last_login_time DESC
            """)
            users = cursor.fetchall()

            return UserListResponse(
                users=users,
                total=len(users)
            )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取用户列表失败: {str(e)}"
        )
    finally:
        conn.close()

@router.get("/admin/logs")
async def get_user_logs(
    current_admin: dict = Depends(get_current_admin_user),
    user_id: Optional[int] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
):
    """获取用户操作日志（仅管理员）"""
    if user_id is None:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"success": False, "message": "user_id is required"},
        )

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    ul.id,
                    ul.user_id,
                    u.username,
                    ul.operate_type,
                    ul.operate_content,
                    ul.operate_ip,
                    ul.created_at
                FROM user_logs ul
                JOIN users u ON u.id = ul.user_id
                WHERE ul.user_id = %s
                ORDER BY ul.created_at DESC
                LIMIT %s
                """,
                (user_id, limit),
            )
            logs = cursor.fetchall()
            return {"success": True, "logs": logs, "total": len(logs)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取日志失败: {str(e)}"
        )
    finally:
        conn.close()

@router.put("/admin/user/{user_id}/password")
async def admin_update_user_password(
    user_id: int,
    request: AdminUpdatePasswordRequest,
    current_admin: dict = Depends(get_current_admin_user)
):
    """管理员修改用户密码"""
    if len(request.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="密码长度至少6个字符"
        )

    conn = get_db_connection()
    try:
        with conn.cursor() as cursor:
            # 检查用户是否存在
            cursor.execute(
                "SELECT id, user_type FROM users WHERE id = %s",
                (user_id,)
            )
            user = cursor.fetchone()

            if not user:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="用户不存在"
                )

            if user['user_type'] == 'admin':
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="无法修改管理员密码"
                )

            # 更新密码
            new_password_hash = hash_password(request.new_password)
            cursor.execute(
                "UPDATE users SET password_hash = %s WHERE id = %s",
                (new_password_hash, user_id)
            )
            conn.commit()

            return {"message": "密码修改成功"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"密码修改失败: {str(e)}"
        )
    finally:
        conn.close()
