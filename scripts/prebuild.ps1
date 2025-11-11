$BuildDir = ".\build"

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