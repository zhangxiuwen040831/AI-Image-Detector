-- 创建数据库
CREATE DATABASE IF NOT EXISTS ai_image_detector_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

USE ai_image_detector_db;

-- 创建用户表
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- 插入测试数据
-- 普通用户
INSERT INTO users (username, password_hash, nickname, user_type, last_login_time, last_logout_time) VALUES
('user1', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4pWGj1BEG6UJvKTO', '测试用户1', 'normal', NOW() - INTERVAL 2 HOUR, NOW() - INTERVAL 1 HOUR),
('user2', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4pWGj1BEG6UJvKTO', '测试用户2', 'normal', NOW() - INTERVAL 1 DAY, NOW() - INTERVAL 23 HOUR),
('testuser', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4pWGj1BEG6UJvKTO', '测试用户', 'normal', NOW() - INTERVAL 30 MINUTE, NULL);

-- 管理员用户
INSERT INTO users (username, password_hash, nickname, user_type, last_login_time, last_logout_time) VALUES
('admin', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/X4pWGj1BEG6UJvKTO', '管理员', 'admin', NOW() - INTERVAL 10 MINUTE, NULL);

-- 说明：密码统一为 'password123' 的bcrypt哈希值
-- 普通用户密码: password123
-- 管理员密码: password123
