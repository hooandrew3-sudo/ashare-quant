# 注册 Windows 计划任务：盘后增量同步 + 每日信号 + 纸面调仓（默认 18:30）
# 用法：powershell -ExecutionPolicy Bypass -File scripts/setup_windows_schedule.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = (Get-Command python).Source

$daily = @"
cd /d "$root" && set PYTHONIOENCODING=utf-8 && "$python" run_pipeline.py --config configs\real.yaml --source baostock --universe csi800 --step daily >> "$root\daily.log" 2>&1
"@

$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $daily"
$trigger = New-ScheduledTaskTrigger -Daily -At 18:30
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Minutes 60)
Register-ScheduledTask -TaskName "AshareQuantDaily" -Action $action -Trigger $trigger -Settings $settings -Description "A股量化：每日数据同步+信号+纸面调仓" -Force
Write-Host "已注册计划任务 AshareQuantDaily（每天 18:30，日志写入 daily.log）"
