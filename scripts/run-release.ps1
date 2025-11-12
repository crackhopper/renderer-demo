. ".\scripts\prebuild.ps1"

Push-Location $BuildDir
try {
    cmake --build . --target VulkanGLFWDemo --config Release
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Release编译成功完成!" -ForegroundColor Green
        
    } else {
        Write-Error "Release编译失败!"
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

## 运行Release版本
& ".\\build\\Release\\VulkanGLFWDemo.exe"