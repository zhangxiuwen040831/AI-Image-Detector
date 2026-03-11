# Git上传步骤指南：从1.0版本升级到2.0版本

## 步骤1：分析项目结构并识别非代码文件

在上传项目之前，需要识别并排除所有非代码文件，包括：

- **模型文件**：`model/`、`checkpoints/`
- **配置文件**：`configs/`（可能包含敏感信息）
- **数据文件**：`data/`、`datasets/`
- **构建产物**：`dist/`、`frontend/dist/`
- **依赖文件**：`node_modules/`、`frontend/node_modules/`
- **日志文件**：`logs/`
- **缓存文件**：`__pycache__/`、`*.pyc`
- **压缩文件**：`*.tar.gz`、`*.zip`
- **IDE相关**：`.idea/`、`.vscode/`
- **临时文件**：`*.tmp`、`*.temp`
- **报告文件**：`jindu.md`、`remote_prepare_report.json`、`dataset_index.json`

## 步骤2：更新.gitignore文件

创建或更新`.gitignore`文件，确保所有非代码文件都被排除：

```gitignore
# IDE related files
.idea/
.vscode/

# Environment and tools
.trae/

# Build outputs
dist/
frontend/dist/

# Logs
logs/

# Model files
model/
checkpoints/

# Cache files
__pycache__/
**/__pycache__/

# Compressed files
*.tar.gz
*.zip
*.rar

# Configuration files (may contain sensitive information)
configs/

# Database related
data_management/

# Data directories
data/
datasets/

# Frontend dependencies
node_modules/
frontend/node_modules/

# Temporary files
*.tmp
*.temp

# Report files
jindu.md
remote_prepare_report.json
dataset_index.json

# OS generated files
Thumbs.db
.DS_Store

# Python bytecode
*.pyc

# Environment variables
.env
.env.local
.env.*.local

# Testing
pytest_cache/

# Coverage
coverage/
.coverage

# Documentation build
docs/_build/
```

## 步骤3：清理缓存文件

删除所有`__pycache__`目录和`.pyc`文件：

```bash
# Windows PowerShell
Get-ChildItem -Path . -Include "__pycache__" -Recurse -Directory | ForEach-Object { Remove-Item -Path $_.FullName -Recurse -Force }
Get-ChildItem -Path . -Include "*.pyc" -Recurse -File | ForEach-Object { Remove-Item -Path $_.FullName -Force }

# Linux/Mac
find . -name "__pycache__" -type d -exec rm -rf {} \;
find . -name "*.pyc" -type f -exec rm -f {} \;
```

## 步骤4：检查git状态

确保只有代码文件被跟踪：

```bash
git status
```

## 步骤5：暂存所有更改

```bash
git add .
```

## 步骤6：提交更改

```bash
git commit -m "Update to version 2.0 with code-only upload"
```

## 步骤7：创建2.0版本标签

```bash
git tag -a v2.0 -m "Version 2.0 release"
```

## 步骤8：推送到远程仓库

```bash
git push origin main
git push origin v2.0
```

## 步骤9：验证上传结果

登录GitHub，检查仓库是否成功更新到2.0版本，确保所有非代码文件都被正确排除。

## 注意事项

1. **版本控制**：使用标签（tag）来标记版本，便于后续回滚和管理。
2. **安全性**：确保敏感配置信息不会被上传到GitHub。
3. **性能**：排除大型文件和目录，减少上传时间和仓库大小。
4. **一致性**：确保`.gitignore`文件在团队成员之间保持一致。

## 故障排除

如果遇到`index.lock`文件锁定问题：

```bash
# 手动删除锁定文件
rm .git/index.lock

# 或使用git命令重置
git reset
```

如果遇到权限问题，确保以管理员身份运行终端。