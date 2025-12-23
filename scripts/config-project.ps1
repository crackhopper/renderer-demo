param (
    [switch]$Force # 定义 -Force 开关参数
)

. ".\scripts\load-dotenv.ps1"

if ($Force) {
    Write-Host "强制刷新配置：清理旧缓存..." -ForegroundColor Yellow
    if (Test-Path "$BuildDir\CMakeCache.txt") {
        Remove-Item "$BuildDir\CMakeCache.txt" -Force
    }
    # 如果你想彻底重新构建，也可以直接删除整个 build 文件夹
    # if (Test-Path $BuildDir) { Remove-Item -Recurse -Force $BuildDir }
}

Write-Host "检查构建目录..."
if (-not (Test-Path "$BuildDir\CMakeCache.txt")) {
    Write-Host "配置CMake项目..." -ForegroundColor Cyan
    if (-not (Test-Path $BuildDir)) {
        New-Item -ItemType Directory -Path $BuildDir -Force | Out-Null
    }
    Push-Location $BuildDir
    try {
        cmake ..
        if ($LASTEXITCODE -ne 0) {
            Write-Error "CMake配置失败!"
            exit $LASTEXITCODE
        }
        Write-Host "CMake配置成功!" -ForegroundColor Green
    }
    finally {
        Pop-Location
    }
} else {
    Write-Host "使用现有CMake配置" -ForegroundColor Green
}