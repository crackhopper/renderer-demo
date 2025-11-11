. ".\scripts\prebuild.ps1"

Push-Location $BuildDir
try {
    cmake --build . --target VulkanGLFWDemo --config Debug
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Debug编译成功完成!" -ForegroundColor Green
        
    } else {
        Write-Error "Debug编译失败!"
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}