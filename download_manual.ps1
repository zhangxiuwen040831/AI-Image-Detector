# NTIRE 2026 数据集手动下载脚本 (PowerShell)
# 使用方法: 在 PowerShell 中运行 .\download_manual.ps1

$baseUrl = "https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train/resolve/main"
$outputDir = "./NTIRE-RobustAIGenDetection-train"

# 创建输出目录
if (!(Test-Path $outputDir)) {
    New-Item -ItemType Directory -Path $outputDir -Force
}

# 文件列表
$files = @(
    @{Name="shard_0.zip"; Size="20.6 GB"},
    @{Name="shard_1.zip"; Size="20.8 GB"},
    @{Name="shard_2.zip"; Size="20.4 GB"},
    @{Name="shard_3.zip"; Size="20.6 GB"},
    @{Name="shard_4.zip"; Size="20.5 GB"},
    @{Name="shard_5.zip"; Size="11.4 GB"}
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "NTIRE 2026 数据集手动下载脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

foreach ($file in $files) {
    $fileName = $file.Name
    $fileSize = $file.Size
    $outputPath = Join-Path $outputDir $fileName
    $url = "$baseUrl/$fileName"
    
    # 检查文件是否已存在且完整
    if (Test-Path $outputPath) {
        $existingSize = (Get-Item $outputPath).Length
        $existingSizeGB = [math]::Round($existingSize / 1GB, 2)
        Write-Host "✅ $fileName 已存在 ($existingSizeGB GB)，跳过下载" -ForegroundColor Green
        continue
    }
    
    Write-Host "📥 正在下载: $fileName ($fileSize)..." -ForegroundColor Yellow
    
    try {
        # 使用 BITS 传输（支持断点续传）
        Import-Module BitsTransfer -ErrorAction SilentlyContinue
        
        if (Get-Command Start-BitsTransfer -ErrorAction SilentlyContinue) {
            # 使用 BITS 下载
            Start-BitsTransfer -Source $url -Destination $outputPath -DisplayName "Downloading $fileName" -Description "NTIRE Dataset"
        } else {
            # 使用 Invoke-WebRequest 下载
            Invoke-WebRequest -Uri $url -OutFile $outputPath -Resume
        }
        
        Write-Host "✅ $fileName 下载完成!" -ForegroundColor Green
    }
    catch {
        Write-Host "❌ $fileName 下载失败: $_" -ForegroundColor Red
        Write-Host "⏳ 继续下载下一个文件..." -ForegroundColor Yellow
    }
    
    Write-Host ""
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "下载完成!" -ForegroundColor Cyan
Write-Host "文件保存在: $((Resolve-Path $outputDir).Path)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
