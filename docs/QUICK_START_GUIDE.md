# AI图像检测系统 - 快速部署指南

## 当前环境状态

✅ **GPU服务器已配置：**
- GPU: NVIDIA GeForce RTX 2080 Ti (11GB)
- CUDA: 12.4
- 驱动: 550.90.07
- PyTorch: 2.6.0+cu124 (已安装)
- 所有依赖: 已安装

✅ **数据已就绪：**
- 训练集: data/genimage/train/ (真实3000+ / 虚假9600+)
- 验证集: data/genimage/val/ (真实333 / 虚假1066)

---

## 方案一：使用云GPU服务器训练（推荐）

### 1. 手动SSH登录并开始训练

在PowerShell中执行以下命令（需要手动输入密码）：

```powershell
ssh -p 57785 root@region-42.seetacloud.com
# 输入密码: BexyrhAT8rbN

# 在服务器上执行:
cd ~/autodl-tmp/ai-image-detector
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-detector

# 设置镜像并开始训练
export HF_ENDPOINT=https://hf-mirror.com
python -m src.training.train_baseline --config configs/cloud_gpu_fast_config.yml
```

### 2. 训练完成后，将模型传输回本地

```powershell
# 在本地PowerShell中执行
scp -P 57785 -r root@region-42.seetacloud.com:~/autodl-tmp/ai-image-detector/checkpoints/demo/* ./checkpoints/demo/
```

---

## 方案二：立即启动Web服务（使用模拟模式）

无需等待训练完成，Web服务已支持模拟模式！

### 1. 激活本地环境

```powershell
.\.venv\Scripts\activate
```

### 2. 启动Web服务

```powershell
python serve_api.py --host 0.0.0.0 --port 8000
```

### 3. 访问Web界面

在浏览器中打开: `web/index.html`

**功能特性（模拟模式）：**
- ✅ 上传图像进行分析
- ✅ 生成AI/真实概率
- ✅ 显示热力图
- ✅ 记录分析历史
- ✅ 训练状态面板

---

## 方案三：启动带用户管理的完整前端（需数据库）

> 适合需要「登录 / 历史记录 / 统计 / 用户设置」等完整功能时使用。

### 1. 准备并启动数据库

1. 在本机或服务器上安装 MySQL 8.0+。  
2. 执行项目自带的完整建表脚本（会创建 `ai_image_detector` 数据库及所有表）：

```bash
mysql -u root -p < database/complete_schema.sql
```

3. 根据实际数据库地址/账号，编辑 `configs/database.yml`，确保与 `database/complete_schema.sql` 中的库名 `ai_image_detector` 一致。

### 2. 启动带认证的 API 服务

在项目根目录激活虚拟环境后运行：

```powershell
.\.venv\Scripts\activate

python serve_api_with_auth.py --host 0.0.0.0 --port 8000
```

该命令会启动 `src.api.server_with_auth`，并开放以下接口供前端使用：

- `/predict`：图像检测（支持热力图）
- `/api/auth/register` / `/api/auth/login` / `/api/auth/logout`：用户注册、登录、退出
- `/api/auth/me` / `/api/auth/settings`：获取与更新当前用户设置
- `/api/history` / `/api/history/{id}` / `/api/history/stats`：检测历史列表、删除、统计

### 3. 启动并使用带用户管理的前端

1. 在本机直接用浏览器打开前端文件：

```text
web/index_with_auth.html
```

2. 打开页面后：

- 顶部右侧点击「登录 / 注册」按钮，先完成注册/登录。  
- 在「🔍 检测」页拖拽或选择图片，前端会调用 `http://localhost:8000/predict` 并叠加热力图。  
- 登录后可切换到：
  - 「📋 历史记录」标签页：查看并删除个人检测记录（数据来自数据库的 `detection_results` 等表）。  
  - 「📊 统计」标签页：查看当前用户的总检测次数、AI/真实数量与占比。  
  - 「⚙️ 设置」标签页：修改通知、自动保存历史、默认视图、主题与语言等（映射到 `user_settings` 表）。

如果前端无法联通后端，请确认：

- 后端监听地址与端口为 `http://localhost:8000`（或按需修改 `web/app_with_auth.js` 中的 `API_BASE`）。  
- 浏览器未被跨域策略拦截（后端已在 `src/api/server_with_auth.py` 中启用 `CORSMiddleware` 允许前端直接访问）。

---

## 完整工作流程记录

### 云GPU配置步骤

1. **连接验证** ✅
   ```
   SSH: ssh -p 57785 root@region-42.seetacloud.com
   GPU: NVIDIA GeForce RTX 2080 Ti
   CUDA: 12.4
   ```

2. **环境配置** ✅
   - Conda环境: ai-detector (Python 3.10)
   - PyTorch: 2.6.0+cu124
   - 所有依赖: 已安装

3. **数据准备** ✅
   - 训练集: 12,600 图像
   - 验证集: 1,399 图像

4. **模型训练** 🔄 (进行中)
   - 配置: configs/cloud_gpu_fast_config.yml
   - 2个epoch，快速验证
   - 使用LoRA微调

5. **模型传输** ⏳ (待完成)
   - 目标: checkpoints/demo/best_epoch_X.pth

6. **Web服务部署** ⏳ (待完成)
   - 使用真实模型推理
   - 完整功能验证

---

## 关键文件说明

| 文件 | 说明 |
|------|------|
| `configs/cloud_gpu_fast_config.yml` | 快速训练配置（2 epoch） |
| `src/models/clip_backbone.py` | 修改后的模型加载（含重试逻辑） |
| `src/api/server.py` | 后端API（支持模拟模式） |
| `web/index.html` | Web前端界面 |

---

## 故障排除

### SSH连接问题
- 使用密码: `BexyrhAT8rbN`
- 端口: 57785

### 模型下载慢
- 已配置HF镜像: `HF_ENDPOINT=https://hf-mirror.com`
- 代码已包含5次重试机制

### Web服务问题
- 确保本地虚拟环境已激活
- 检查端口8000未被占用
