-- 1. 创建数据库
CREATE DATABASE IF NOT EXISTS ai_image_detector
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE ai_image_detector;

-- ============================================
-- 1. 用户表 (users) - 优化：添加了最后活动时间
-- ============================================
CREATE TABLE IF NOT EXISTS users (
    id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) NOT NULL UNIQUE COMMENT '用户名',
    email VARCHAR(100) NOT NULL UNIQUE COMMENT '邮箱',
    password_hash VARCHAR(255) NOT NULL COMMENT '密码哈希',
    full_name VARCHAR(100) NULL COMMENT '姓名',
    avatar_url VARCHAR(500) NULL COMMENT '头像URL',
    role ENUM('user', 'admin', 'superadmin') DEFAULT 'user' COMMENT '用户角色',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否激活',
    last_login_at TIMESTAMP NULL COMMENT '最后登录时间',
    last_active_at TIMESTAMP NULL COMMENT '最后活动时间', -- 新增
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    INDEX idx_user_credential (username, email), -- 复合索引优化登录查询
    INDEX idx_role_active (role, is_active),
    INDEX idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户表';

-- ============================================
-- 2. 用户会话表 (user_sessions) - 优化：添加了最后活跃时间
-- ============================================
CREATE TABLE IF NOT EXISTS user_sessions (
    id VARCHAR(100) PRIMARY KEY COMMENT '会话ID',
    user_id INT NOT NULL COMMENT '用户ID',
    token VARCHAR(500) NOT NULL UNIQUE COMMENT '认证Token',
    refresh_token VARCHAR(500) NULL COMMENT '刷新Token', -- 新增，用于无感刷新
    ip_address VARCHAR(45) NULL COMMENT 'IP地址',
    user_agent VARCHAR(500) NULL COMMENT '用户代理',
    last_activity_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '最后活动时间', -- 新增
    expires_at TIMESTAMP NOT NULL COMMENT '过期时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_user_id_expires (user_id, expires_at), -- 复合索引优化清理查询
    INDEX idx_token (token(100)), -- 前缀索引优化
    INDEX idx_expires_at (expires_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户会话表';

-- ============================================
-- 3. 图像表 (images) - 优化：添加了图片元数据存储
-- ============================================
CREATE TABLE IF NOT EXISTS images (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '图像ID',
    user_id INT NULL COMMENT '上传用户ID',
    original_filename VARCHAR(255) NOT NULL COMMENT '原始文件名',
    stored_filename VARCHAR(255) NOT NULL UNIQUE COMMENT '存储文件名（唯一）', -- 改为UNIQUE
    storage_path VARCHAR(500) NOT NULL COMMENT '存储路径',
    file_size BIGINT NOT NULL COMMENT '文件大小(字节)',
    mime_type VARCHAR(100) NOT NULL COMMENT 'MIME类型',
    image_width INT NULL COMMENT '图像宽度',
    image_height INT NULL COMMENT '图像高度',
    metadata JSON NULL COMMENT 'EXIF等元数据（JSON格式）', -- 新增
    upload_source VARCHAR(50) DEFAULT 'web' COMMENT '上传来源',
    md5_hash VARCHAR(32) NULL COMMENT '文件MD5',
    sha256_hash VARCHAR(64) NULL UNIQUE COMMENT '文件SHA256（唯一）',
    is_deleted BOOLEAN DEFAULT FALSE COMMENT '是否删除',
    deleted_at TIMESTAMP NULL COMMENT '删除时间', -- 新增
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_user_created (user_id, created_at DESC), -- 优化用户历史查询
    INDEX idx_hash (sha256_hash), -- 已为UNIQUE，索引自动创建
    INDEX idx_deleted_created (is_deleted, created_at) -- 优化清理查询
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='图像表';

-- ============================================
-- 4. 模型信息表 (models) - 已优化
-- ============================================
CREATE TABLE IF NOT EXISTS models (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '模型ID',
    name VARCHAR(100) NOT NULL COMMENT '模型名称',
    version VARCHAR(50) NOT NULL COMMENT '版本号',
    description TEXT NULL COMMENT '模型描述',
    model_type VARCHAR(50) NOT NULL COMMENT '模型类型',
    checkpoint_path VARCHAR(500) NULL COMMENT '权重文件路径',
    backbone_name VARCHAR(100) NULL COMMENT '骨干网络名称',
    input_size INT DEFAULT 224 COMMENT '输入尺寸',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否激活',
    is_default BOOLEAN DEFAULT FALSE COMMENT '是否默认模型',
    accuracy FLOAT NULL COMMENT '准确率',
    `precision` FLOAT NULL COMMENT '精确率', -- 反引号包裹保留字
    recall FLOAT NULL COMMENT '召回率',
    f1_score FLOAT NULL COMMENT 'F1分数',
    training_date DATE NULL COMMENT '训练日期',
    created_by INT NULL COMMENT '创建用户ID',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (created_by) REFERENCES users(id) ON DELETE SET NULL,
    UNIQUE INDEX idx_name_version (name, version), -- 确保名称+版本唯一
    INDEX idx_active_default (is_active, is_default)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='模型信息表';

-- ============================================
-- 5. 检测结果表 (detection_results) - 优化：添加了性能索引
-- ============================================
CREATE TABLE IF NOT EXISTS detection_results (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '结果ID',
    image_id BIGINT NOT NULL COMMENT '图像ID',
    user_id INT NULL COMMENT '用户ID',
    model_id INT NULL COMMENT '使用的模型ID',
    detection_status ENUM('pending', 'processing', 'completed', 'failed') DEFAULT 'pending' COMMENT '检测状态',
    is_fake BOOLEAN NULL COMMENT '是否为AI生成',
    confidence_score DECIMAL(5,4) NULL COMMENT '置信度(0.0000-1.0000)', -- 改用DECIMAL更精确
    prediction_probability DECIMAL(5,4) NULL COMMENT '预测概率',
    explanation TEXT NULL COMMENT '检测说明',
    heatmap_data JSON NULL COMMENT '热力图数据',
    raw_result JSON NULL COMMENT '原始检测结果',
    processing_time_ms INT NULL COMMENT '处理时间(毫秒)',
    error_message TEXT NULL COMMENT '错误信息',
    detected_at TIMESTAMP NULL COMMENT '检测完成时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE SET NULL,
    UNIQUE INDEX idx_unique_image_detection (image_id, model_id), -- 防止同一图片重复检测
    INDEX idx_user_status (user_id, detection_status, created_at DESC), -- 优化用户查询
    INDEX idx_fake_confidence (is_fake, confidence_score DESC), -- 优化统计查询
    INDEX idx_created_at_status (created_at, detection_status) -- 优化后台管理
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='检测结果表';

-- ============================================
-- 6. 任务队列表 (task_queue) - 优化：添加了结果关联
-- ============================================
CREATE TABLE IF NOT EXISTS task_queue (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '任务ID',
    image_id BIGINT NOT NULL COMMENT '图像ID',
    result_id BIGINT NULL COMMENT '关联的结果ID', -- 新增，直接关联结果
    user_id INT NULL COMMENT '提交用户ID',
    model_id INT NULL COMMENT '使用的模型ID',
    task_status ENUM('queued', 'processing', 'completed', 'failed', 'cancelled') DEFAULT 'queued' COMMENT '任务状态',
    priority TINYINT DEFAULT 5 COMMENT '优先级(1-10)', -- 改为TINYINT节省空间
    retry_count TINYINT DEFAULT 0 COMMENT '重试次数',
    max_retries TINYINT DEFAULT 3 COMMENT '最大重试次数',
    error_message TEXT NULL COMMENT '错误信息',
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '提交时间',
    started_at TIMESTAMP NULL COMMENT '开始时间',
    completed_at TIMESTAMP NULL COMMENT '完成时间',
    FOREIGN KEY (image_id) REFERENCES images(id) ON DELETE CASCADE,
    FOREIGN KEY (result_id) REFERENCES detection_results(id) ON DELETE SET NULL, -- 新增外键
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    FOREIGN KEY (model_id) REFERENCES models(id) ON DELETE SET NULL,
    INDEX idx_status_priority (task_status, priority, submitted_at), -- 优化任务获取
    INDEX idx_user_image (user_id, image_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='任务队列表';

-- ============================================
-- 7. 用户设置表 (user_settings) - 无重大变更
-- ============================================
CREATE TABLE IF NOT EXISTS user_settings (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '设置ID',
    user_id INT NOT NULL UNIQUE COMMENT '用户ID',
    notification_enabled BOOLEAN DEFAULT TRUE COMMENT '启用通知',
    auto_save_history BOOLEAN DEFAULT TRUE COMMENT '自动保存历史',
    default_view_mode VARCHAR(20) DEFAULT 'grid' COMMENT '默认视图',
    theme VARCHAR(20) DEFAULT 'light' COMMENT '主题',
    language VARCHAR(10) DEFAULT 'zh-CN' COMMENT '语言',
    default_model_id INT NULL COMMENT '默认使用的模型',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT '更新时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    FOREIGN KEY (default_model_id) REFERENCES models(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='用户设置表';

-- ============================================
-- 8. 系统日志表 (system_logs) - 优化：添加日志级别
-- ============================================
CREATE TABLE IF NOT EXISTS system_logs (
    id BIGINT AUTO_INCREMENT PRIMARY KEY COMMENT '日志ID',
    user_id INT NULL COMMENT '用户ID',
    log_level ENUM('debug', 'info', 'warn', 'error', 'fatal') DEFAULT 'info' COMMENT '日志级别', -- 新增
    action VARCHAR(100) NOT NULL COMMENT '操作类型',
    endpoint VARCHAR(500) NULL COMMENT 'API端点', -- 新增
    ip_address VARCHAR(45) NULL COMMENT 'IP地址',
    user_agent VARCHAR(500) NULL COMMENT '用户代理',
    request_data JSON NULL COMMENT '请求数据',
    response_data JSON NULL COMMENT '响应数据',
    status_code INT NULL COMMENT 'HTTP状态码',
    error_message TEXT NULL COMMENT '错误信息',
    execution_time_ms INT NULL COMMENT '执行时间(毫秒)',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    INDEX idx_level_created (log_level, created_at DESC), -- 优化日志查询
    INDEX idx_action_user (action, user_id, created_at DESC)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='系统日志表';

-- ============================================
-- 9. 新增：API密钥表 (api_keys) - 用于第三方接入
-- ============================================
CREATE TABLE IF NOT EXISTS api_keys (
    id INT AUTO_INCREMENT PRIMARY KEY COMMENT '密钥ID',
    user_id INT NOT NULL COMMENT '所属用户ID',
    name VARCHAR(100) NOT NULL COMMENT '密钥名称',
    api_key VARCHAR(64) NOT NULL UNIQUE COMMENT 'API密钥',
    api_secret VARCHAR(128) NULL COMMENT 'API密钥（加密存储）',
    permissions JSON NULL COMMENT '权限配置',
    rate_limit_per_hour INT DEFAULT 100 COMMENT '每小时限制',
    is_active BOOLEAN DEFAULT TRUE COMMENT '是否激活',
    last_used_at TIMESTAMP NULL COMMENT '最后使用时间',
    expires_at TIMESTAMP NULL COMMENT '过期时间',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP COMMENT '创建时间',
    FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
    INDEX idx_api_key (api_key),
    INDEX idx_user_active (user_id, is_active)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='API密钥表';

-- ============================================
-- 初始化数据
-- ============================================
-- 插入默认模型
INSERT INTO models (name, version, description, model_type, backbone_name, input_size, is_default, is_active, accuracy, `precision`, recall, f1_score, training_date, created_at)
VALUES (
    'ViT-L-14 OpenCLIP',
    '1.0.0',
    '基于OpenCLIP的ViT-Large-14模型，用于AI图像检测',
    'clip_vitl14',
    'ViT-L-14',
    224,
    TRUE,
    TRUE,
    0.9823,
    0.9678,
    0.9541,
    0.9609,
    '2024-11-15',
    NOW()
) ON DUPLICATE KEY UPDATE updated_at = NOW();

-- 插入备用模型
INSERT INTO models (name, version, description, model_type, backbone_name, input_size, is_default, is_active, accuracy, `precision`, recall, f1_score, training_date, created_at)
VALUES (
    'ConvNeXt-Base',
    '1.1.0',
    '基于ConvNeXt架构的专用检测模型',
    'convnext_base',
    'ConvNeXt-B',
    384,
    FALSE,
    TRUE,
    0.9756,
    0.9612,
    0.9487,
    0.9549,
    '2024-12-01',
    NOW()
) ON DUPLICATE KEY UPDATE updated_at = NOW();

-- ============================================
-- 创建视图：简化常用查询
-- ============================================
-- 视图1：检测结果概览
CREATE OR REPLACE VIEW v_detection_overview AS
SELECT
    dr.id,
    dr.image_id,
    i.original_filename,
    i.stored_filename,
    u.username,
    m.name as model_name,
    dr.detection_status,
    dr.is_fake,
    dr.confidence_score,
    dr.processing_time_ms,
    dr.created_at
FROM detection_results dr
LEFT JOIN images i ON dr.image_id = i.id
LEFT JOIN users u ON dr.user_id = u.id
LEFT JOIN models m ON dr.model_id = m.id
WHERE i.is_deleted = FALSE;

-- 视图2：用户活动统计
CREATE OR REPLACE VIEW v_user_activity AS
SELECT
    u.id as user_id,
    u.username,
    u.email,
    u.role,
    u.last_login_at,
    COUNT(DISTINCT i.id) as total_images,
    COUNT(DISTINCT dr.id) as total_detections,
    MAX(dr.created_at) as last_detection_time
FROM users u
LEFT JOIN images i ON u.id = i.user_id
LEFT JOIN detection_results dr ON u.id = dr.user_id
GROUP BY u.id, u.username, u.email, u.role, u.last_login_at;

-- ============================================
-- 创建事件：定期清理过期数据
-- ============================================
-- 启用事件调度器（需MySQL有EVENT权限）
SET GLOBAL event_scheduler = ON;

-- 事件：每天清理30天前的过期会话
DELIMITER $$
CREATE EVENT IF NOT EXISTS event_cleanup_old_sessions
ON SCHEDULE EVERY 1 DAY
STARTS CURRENT_TIMESTAMP
DO
BEGIN
    DELETE FROM user_sessions WHERE expires_at < NOW() - INTERVAL 1 DAY;
    DELETE FROM task_queue WHERE task_status IN ('completed', 'failed', 'cancelled') AND completed_at < NOW() - INTERVAL 7 DAY;
END
$$
DELIMITER ;

-- ============================================
-- 创建完成
-- ============================================
SELECT '✅ 数据库表结构创建完成！' AS message;
SELECT '📊 已创建 9 张核心表' AS tables_summary;
SELECT '👤 users - 用户表' AS table_1;
SELECT '🔐 user_sessions - 用户会话表' AS table_2;
SELECT '🖼️ images - 图像存储表' AS table_3;
SELECT '🤖 models - AI模型表' AS table_4;
SELECT '📈 detection_results - 检测结果表' AS table_5;
SELECT '📋 task_queue - 任务队列表' AS table_6;
SELECT '⚙️ user_settings - 用户设置表' AS table_7;
SELECT '📊 system_logs - 系统日志表' AS table_8;
SELECT '🔑 api_keys - API密钥表（新增）' AS table_9;
SELECT '👁️ 已创建 2 个视图：v_detection_overview, v_user_activity' AS views_summary;
SELECT '🔄 已创建 1 个定时清理事件' AS events_summary;
