# Tesseract OCR 安装指南

## 方法一：使用winget安装（推荐）

打开PowerShell（管理员权限），运行：

```powershell
winget install Tesseract-OCR.Tesseract --accept-source-agreements --accept-package-agreements
```

## 方法二：手动下载安装

### 步骤1：下载安装程序

访问以下链接下载Tesseract：
https://github.com/UB-Mannheim/tesseract-wiki/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe

### 步骤2：运行安装程序

1. 双击下载的 `tesseract-ocr-w64-setup-5.5.0.20241111.exe`
2. 按照安装向导完成安装
3. **重要**：记住安装路径（通常是 `C:\Program Files\Tesseract-OCR`）

### 步骤3：配置环境变量

1. 右键点击"此电脑" → "属性"
2. 点击"高级系统设置"
3. 点击"环境变量"
4. 在"系统变量"中找到"Path"，点击"编辑"
5. 点击"新建"，添加Tesseract安装路径：
   - `C:\Program Files\Tesseract-OCR`
6. 点击"确定"保存所有设置

### 步骤4：验证安装

打开新的PowerShell窗口，运行：

```powershell
tesseract --version
```

如果显示版本号（如 `tesseract 5.5.0`），说明安装成功！

## 方法三：使用chocolatey安装

如果您安装了chocolatey，可以运行：

```powershell
choco install tesseract
```

## 安装后

1. **重启系统**：关闭并重新启动知识追踪系统
2. **测试功能**：进入"上传数据" → "拍照/图片上传"
3. **上传图片**：选择一张包含题目的图片
4. **查看结果**：系统会自动识别图片中的文字

## 常见问题

### Q: 安装后还是提示缺少库？
A: 确保重启了系统，并且配置了环境变量

### Q: 识别效果不好？
A: 确保图片清晰、光线充足、文字完整

### Q: 只识别英文？
A: 需要下载中文语言包，放到Tesseract的tessdata目录

## 中文语言包下载

如果需要识别中文，下载以下文件：
https://github.com/tesseract-ocr/tessdata/raw/main/chi_sim.traineddata

将下载的文件放到：
`C:\Program Files\Tesseract-OCR\tessdata\`

## 联系支持

如果遇到问题，请访问：
https://github.com/UB-Mannheim/tesseract/wiki
