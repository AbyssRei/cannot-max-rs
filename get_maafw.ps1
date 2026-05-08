$repo = 'MaaXYZ/MaaFramework'
$apiUrl = "https://api.github.com/repos/$repo/releases/latest"

# 1. 获取最新版本号
Write-Host "Fetching latest release info..."
try {
    $tag = (Invoke-RestMethod -Uri $apiUrl).tag_name
} catch {
    Write-Host "Failed to fetch GitHub API. Error: $_" -ForegroundColor Red
    return
}

$folderName = "MAA-win-x86_64-$tag"
$targetPath = Join-Path (Get-Location) $folderName
$zipPath = Join-Path (Get-Location) "$folderName.zip"

# 2. 检查是否已是最新版
if (Test-Path $targetPath) {
    Write-Host "MaaFramework is already the latest version ($tag). No update needed." -ForegroundColor Green
    return
}

# 3. 不是最新版，删除旧版目录
$oldFolders = Get-ChildItem -Directory -Filter "MAA-win-x86_64-*"
foreach ($folder in $oldFolders) {
    Write-Host "Found old version: $($folder.Name). Removing..." -ForegroundColor Yellow
    Remove-Item -Path $folder.FullName -Recurse -Force
}

# 4. 下载最新版 (带进度条)
Write-Host "Downloading latest version $tag..."
$downloadUrl = "https://v6.gh-proxy.org/https://github.com/$repo/releases/download/$tag/$folderName.zip"

$wc = New-Object System.Net.WebClient
$Global:DownloadComplete = $false
$Global:DownloadPercentage = 0

# 注册进度变化事件
Register-ObjectEvent -InputObject $wc -EventName DownloadProgressChanged -SourceIdentifier WcDownloadProgress -Action {
    $Global:DownloadPercentage = $EventArgs.ProgressPercentage
} | Out-Null

# 注册下载完成事件
Register-ObjectEvent -InputObject $wc -EventName DownloadFileCompleted -SourceIdentifier WcDownloadComplete -Action {
    $Global:DownloadComplete = $true
} | Out-Null

try {
    # 异步下载
    $wc.DownloadFileAsync($downloadUrl, $zipPath)
    
    # 循环显示进度条
    while (-not $Global:DownloadComplete) {
        Write-Progress -Activity "Downloading $folderName.zip" -Status "Progress:" -PercentComplete $Global:DownloadPercentage -CurrentOperation "$Global:DownloadPercentage%"
        Start-Sleep -Milliseconds 200
    }
    # 清理进度条
    Write-Progress -Activity "Downloading $folderName.zip" -Completed
    
    # 检查是否因为错误而完成
    if (-not (Test-Path $zipPath) -or (Get-Item $zipPath).Length -eq 0) {
        throw "Download failed or file is empty."
    }
} catch {
    Write-Host "Download failed! Error: $_" -ForegroundColor Red
    return
} finally {
    # 注销事件并释放资源
    Unregister-Event -SourceIdentifier WcDownloadProgress -ErrorAction SilentlyContinue
    Unregister-Event -SourceIdentifier WcDownloadComplete -ErrorAction SilentlyContinue
    $wc.Dispose()
}

# 5. 解压到同名子目录
Write-Host "Expanding archive to $folderName..."
# 使用 Join-Path 显式指定解压到名为 $folderName 的子文件夹中
Expand-Archive -Path $zipPath -DestinationPath (Join-Path (Get-Location) $folderName) -Force

# 6. 清理 ZIP 文件
Remove-Item -Path $zipPath -Force
Write-Host "Update/Install complete! Saved to: $targetPath" -ForegroundColor Green
