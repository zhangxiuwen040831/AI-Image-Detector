#!/bin/bash

# 自动化解压脚本
# 功能：在指定时间后对archive(2).zip文件执行解压操作

# 配置参数
TARGET_DIR="/root/lanyun-tmp/ai-image-detector"
ZIP_FILE="archive(2).zip"
DELAY_SECONDS=7200  # 2小时
CHECK_INTERVAL=600  # 10分钟
MAX_CHECKS=12       # 最多检查12次（2小时）

# 日志函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 检查文件是否存在且完整
check_file() {
    if [ -f "$TARGET_DIR/$ZIP_FILE" ]; then
        log "检测到文件 $ZIP_FILE，检查文件大小..."
        # 获取文件大小（字节）
        FILE_SIZE=$(stat -c "%s" "$TARGET_DIR/$ZIP_FILE" 2>/dev/null)
        if [ $? -eq 0 ] && [ "$FILE_SIZE" -gt 0 ]; then
            log "文件大小: $FILE_SIZE 字节，文件存在且完整"
            return 0
        else
            log "文件存在但可能不完整"
            return 1
        fi
    else
        log "文件 $ZIP_FILE 不存在"
        return 1
    fi
}

# 执行解压操作
execute_unzip() {
    log "开始执行解压操作..."
    cd "$TARGET_DIR" || { log "无法进入目录 $TARGET_DIR"; exit 1; }
    
    if [ -f "$ZIP_FILE" ]; then
        log "正在解压 $ZIP_FILE..."
        unzip "$ZIP_FILE" -d .
        if [ $? -eq 0 ]; then
            log "解压完成！"
            # 验证解压结果
            if [ "$(ls -la | grep -v "$ZIP_FILE" | grep -v "^d" | wc -l)" -gt 0 ]; then
                log "解压成功，文件已提取"
            else
                log "解压可能失败，未发现提取的文件"
            fi
        else
            log "解压失败"
        fi
    else
        log "错误：文件 $ZIP_FILE 不存在"
    fi
}

# 主函数
main() {
    log "自动化解压脚本启动"
    log "目标目录: $TARGET_DIR"
    log "目标文件: $ZIP_FILE"
    log "延迟时间: $DELAY_SECONDS 秒 (2小时)"
    
    # 等待指定时间
    log "开始等待 $DELAY_SECONDS 秒..."
    sleep $DELAY_SECONDS
    log "等待完成，开始检查文件..."
    
    # 检查文件是否完整，若不完整则每隔10分钟检查一次
    check_count=0
    while [ $check_count -lt $MAX_CHECKS ]; do
        if check_file; then
            break
        else
            check_count=$((check_count + 1))
            if [ $check_count -lt $MAX_CHECKS ]; then
                log "文件不完整，$CHECK_INTERVAL 秒后再次检查..."
                sleep $CHECK_INTERVAL
            else
                log "已达到最大检查次数，文件仍不完整，停止检查"
                exit 1
            fi
        fi
    done
    
    # 执行解压
    execute_unzip
    
    log "脚本执行完成"
}

# 运行主函数
main
