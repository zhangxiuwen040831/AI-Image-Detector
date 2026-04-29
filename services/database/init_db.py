import pymysql
import sys
import os

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'user': 'root',
    'password': '123456',
    'charset': 'utf8mb4',
    'cursorclass': pymysql.cursors.DictCursor
}

def init_database():
    """初始化数据库"""
    try:
        # 连接MySQL服务器
        print("正在连接MySQL服务器...")
        connection = pymysql.connect(**DB_CONFIG)

        try:
            with connection.cursor() as cursor:
                # 创建数据库
                print("创建数据库 ai_image_detector_db...")
                cursor.execute("CREATE DATABASE IF NOT EXISTS ai_image_detector_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
                cursor.execute("USE ai_image_detector_db")

                # 创建用户表
                print("创建用户表 users...")
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id INT AUTO_INCREMENT PRIMARY KEY COMMENT '用户ID',
                        username VARCHAR(100) NOT NULL UNIQUE COMMENT '用户名',
                        password_hash VARCHAR(255) NOT NULL COMMENT '密码哈希',
                        nickname VARCHAR(100) NOT NULL DEFAULT '用户' COMMENT '昵称',
                        user_type ENUM('normal', 'admin') NOT NULL DEFAULT 'normal' COMMENT '用户类型：normal普通用户，admin管理员',
                        last_login_time DATETIME DEFAULT NULL COMMENT '最近登录时间',
                        last_logout_time DATETIME DEFAULT NULL COMMENT '最近登出时间',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
                        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
                        is_active BOOLEAN DEFAULT TRUE COMMENT '是否激活',
                        INDEX idx_username (username),
                        INDEX idx_user_type (user_type),
                        INDEX idx_last_login (last_login_time)
                    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表'
                """)

                # 插入测试数据
                print("插入测试数据...")

                # 先删除已存在的测试用户
                cursor.execute("DELETE FROM users WHERE username IN ('user1', 'user2', 'testuser', 'admin')")

                # 导入bcrypt用于密码哈希
                try:
                    import bcrypt
                    password_hash = bcrypt.hashpw('password123'.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                except ImportError:
                    # 如果bcrypt不可用，使用固定的哈希值
                    password_hash = '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4pWGj1BEG6UJvKTO'

                # 插入普通用户
                cursor.execute("""
                    INSERT INTO users (username, password_hash, nickname, user_type, last_login_time, last_logout_time)
                    VALUES ('user1', %s, '测试用户1', 'normal', DATE_SUB(NOW(), INTERVAL 2 HOUR), DATE_SUB(NOW(), INTERVAL 1 HOUR))
                """, (password_hash,))

                cursor.execute("""
                    INSERT INTO users (username, password_hash, nickname, user_type, last_login_time, last_logout_time)
                    VALUES ('user2', %s, '测试用户2', 'normal', DATE_SUB(NOW(), INTERVAL 1 DAY), DATE_SUB(NOW(), INTERVAL 23 HOUR))
                """, (password_hash,))

                cursor.execute("""
                    INSERT INTO users (username, password_hash, nickname, user_type, last_login_time, last_logout_time)
                    VALUES ('testuser', %s, '测试用户', 'normal', DATE_SUB(NOW(), INTERVAL 30 MINUTE), NULL)
                """, (password_hash,))

                # 插入管理员用户
                cursor.execute("""
                    INSERT INTO users (username, password_hash, nickname, user_type, last_login_time, last_logout_time)
                    VALUES ('admin', %s, '管理员', 'admin', DATE_SUB(NOW(), INTERVAL 10 MINUTE), NULL)
                """, (password_hash,))

                connection.commit()

                # 验证数据
                print("\n验证数据插入结果:")
                cursor.execute("SELECT id, username, nickname, user_type, last_login_time, last_logout_time FROM users")
                users = cursor.fetchall()
                for user in users:
                    print(f"  - {user['username']} ({user['user_type']}): {user['nickname']}")

                print("\n数据库初始化成功！")
                print("测试账号信息:")
                print("  普通用户: user1 / password123")
                print("  普通用户: user2 / password123")
                print("  普通用户: testuser / password123")
                print("  管理员:   admin / password123")

        finally:
            connection.close()

    except pymysql.err.OperationalError as e:
        print(f"数据库连接失败: {e}")
        print("请确保MySQL服务已启动，并且用户名密码正确")
        sys.exit(1)
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()
