# load-dotenv.ps1
param(
    [string]$Path = ".env"
)

if (-Not (Test-Path $Path)) {
    Write-Warning ".env file not found at $Path"
    return
}

# 用来记录已经加载的 key=value
$LoadedVars = @{}

Get-Content $Path | ForEach-Object {
    $line = $_.Trim()
    # 跳过空行或注释
    if ($line -eq "" -or $line.StartsWith("#")) { return }

    # 分割 key=value
    $parts = $line -split "=", 2
    if ($parts.Count -eq 2) {
        $key = $parts[0].Trim()
        $value = $parts[1].Trim().Trim('"') # 去掉可选引号

        # 设置全局变量
        Set-Variable -Name $key -Value $value -Scope Global

        # 记录已加载
        $LoadedVars[$key] = $value
    }
}

# 检查 PrintEnv 是否为 1
if ($LoadedVars.ContainsKey("PrintEnv") -and $LoadedVars["PrintEnv"] -eq "1") {
    Write-Host "=== Loaded .env variables ==="
    foreach ($k in $LoadedVars.Keys) {
        Write-Host "$k = $($LoadedVars[$k])"
    }
    Write-Host "============================="
}
