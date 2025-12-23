. ".\scripts\config-project.ps1"

Push-Location $BuildDir
try {
    cmake --build . --target $ProjectName --config $BuildType
    if ($LASTEXITCODE -eq 0) {
        Write-Host "$BuildType编译成功完成!" -ForegroundColor Green
        
    } else {
        Write-Error "$BuildType编译失败!"
        exit $LASTEXITCODE
    }
}
finally {
    Pop-Location
}