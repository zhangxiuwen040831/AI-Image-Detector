# 用户管理系统 - 完整设置指南

## 📋 概述

本文档说明如何在现有AI图像检测系统基础上添加用户管理功能。

## 🗂️ 新增文件结构

```
ai-image-detector/
├── database/
│   └── schema.sql                 # MySQL数据库建表语句
├── configs/
│   └── database.yml               # 数据库配置
├── src/
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py              # 数据模型
│   │   └── db.py                  # 数据库操作
│   └── auth/
│       ├── __init__.py
│       └── auth_service.py        # 认证服务
└── USER_MANAGEMENT_SETUP_GUIDE.md # 本文档
```

---

## 🚀 快速开始

### 1. 安装新依赖

```bash
pip install pymysql bcrypt python-jose[cryptography]
```

或者直接安装：
```bash
pip install -r requirements.txt
```

### 2. 配置MySQL数据库

#### 2.1 安装MySQL（如果未安装）

**Windows:**
- 下载 MySQL Community Server: https://dev.mysql.com/downloads/mysql/
- 安装并记住root密码

**Linux (Ubuntu/Debian):**
```bash
sudo apt update
sudo apt install mysql-server
sudo mysql_secure_installation
```

#### 2.2 创建数据库和表

```bash
mysql -u root -p < database/schema.sql
```

或者手动执行：
```sql
source database/schema.sql
```

### 3. 配置数据库连接

编辑 `configs/database.yml`：

```yaml
database:
  host: localhost
  port: 3306
  username: root
  password: your_actual_password_here  # 改成你的MySQL密码
  database: ai_image_detector
  # ...
```

### 4. 修改session密钥

在 `configs/database.yml` 中修改：
```yaml
session:
  secret_key: change-this-to-a-random-secret-key-in-production  # 修改为随机密钥
```

---

## 📊 API接口说明

### 认证接口（需要添加到server.py）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/auth/register` | 用户注册 |
| POST | `/api/auth/login` | 用户登录 |
| POST | `/api/auth/logout` | 用户登出 |
| GET | `/api/auth/me` | 获取当前用户信息 |
| GET | `/api/auth/settings` | 获取用户设置 |
| PUT | `/api/auth/settings` | 更新用户设置 |

### 历史记录接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/history` | 获取检测历史 |
| DELETE | `/api/history/{id}` | 删除历史记录 |
| GET | `/api/history/stats` | 获取统计信息 |

---

## 🔒 安全特性

1. **密码加密**: 使用bcrypt进行密码哈希
2. **会话管理**: JWT风格的签名token
3. **CSRF防护**: 支持CSRF token
4. **XSS防护**: 输入验证和输出编码
5. **SQL注入防护**: 使用参数化查询

---

## 📝 注意事项

1. **生产环境部署**：
   - 修改 `secret_key` 为强随机密钥
   - 启用HTTPS
   - 配置防火墙规则
   - 定期备份数据库

2. **数据库优化**：
   - 根据实际负载调整连接池大小
   - 定期清理过期会话
   - 监控慢查询

---

## 🎯 下一步

要完整实现用户界面和API集成，需要：
1. 更新 `src/api/server.py` 添加认证API
2. 更新前端 `web/index.html` 添加登录/注册界面
3. 实现检测历史记录功能
4. 添加用户个人中心页面
