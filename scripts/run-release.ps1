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

$env:VK_LOADER_LAYERS_DISABLE="~implicit~"
$env:VK_DRIVER_FILES="C:\WINDOWS\System32\DriverStore\FileRepository\nvmi.inf_amd64_c6ae241e95feb82d\nv-vk64.json"
## 运行Release版本
& ".\\build\\Release\\VulkanGLFWDemo.exe"