<#
.SYNOPSIS
  Установка агента photo-restore на ПК с Windows 11 (ТЗ §7.1).

.DESCRIPTION
  1. Создаёт venv в agent\.venv (Python 3.12) и ставит в него пакет restore-agent.
  2. Копирует config.example.toml в config.toml, если его ещё нет, и оставляет права на файл
     только текущему пользователю (в нём токен).
  3. Регистрирует задачу Планировщика «при входе в систему»: pythonw.exe, без окна консоли.

  Запуск из любой папки:
    powershell -ExecutionPolicy Bypass -File agent\install.ps1
  Ставит агента из этой папки (рабочей копии). Повторный запуск обновляет пакет и задачу и
  перезапускает работающего агента, config.toml не трогает. Обновлять агента до сборки,
  которая работает на сервере, удобнее agent\update.ps1: он ставит её с GitHub.
#>
param(
    [string]$TaskName = "photo-restore agent"
)
$ErrorActionPreference = "Stop"

$Here = $PSScriptRoot
$Venv = Join-Path $Here ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Pythonw = Join-Path $Venv "Scripts\pythonw.exe"
$Config = Join-Path $Here "config.toml"

if (-not (Test-Path $Python)) {
    Write-Host "Создаю venv: $Venv"
    if (Get-Command py -ErrorAction SilentlyContinue) {
        & py -3.12 -m venv $Venv
    } else {
        & python -m venv $Venv
    }
    if ($LASTEXITCODE -ne 0) { throw "Не удалось создать venv: нужен Python 3.12" }
}

Write-Host "Ставлю restore-agent"
& $Python -m pip install --quiet --upgrade pip
& $Python -m pip install --quiet --upgrade $Here
if ($LASTEXITCODE -ne 0) { throw "pip install не удался" }

if (-not (Test-Path $Config)) {
    Copy-Item (Join-Path $Here "config.example.toml") $Config
    Write-Host "Создан $Config — заполните server_url, token и agent_id" -ForegroundColor Yellow
}
# только текущий пользователь: наследование выключено, остальные права сняты
icacls $Config /inheritance:r /grant:r "$($env:USERNAME):(R,W)" | Out-Null

$Action = New-ScheduledTaskAction -Execute $Pythonw `
    -Argument "-m restore_agent --config `"$Config`"" -WorkingDirectory $Here
$Trigger = New-ScheduledTaskTrigger -AtLogOn -User "$env:USERDOMAIN\$env:USERNAME"
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)
$Running = (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue).State -eq "Running"
Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Settings $Settings `
    -Description "Агент photo-restore: берёт задачи с сервера и выполняет их в ComfyUI" `
    -Force | Out-Null

Write-Host "Задача «$TaskName» зарегистрирована и стартует при входе в систему."
if ($Running) {
    # работающий агент выполняет старый код — перезапускаем
    Stop-ScheduledTask -TaskName $TaskName
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Агент перезапущен с новой версией."
} else {
    Write-Host "Запустить сейчас:   Start-ScheduledTask -TaskName '$TaskName'"
}
Write-Host "Проверить вручную:  & '$Python' -m restore_agent --config '$Config'"
Write-Host "Лог:                %LOCALAPPDATA%\restore-agent\agent.log"
Write-Host "Обновление:         powershell -ExecutionPolicy Bypass -File agent\update.ps1"
Write-Host "config.toml можно править на ходу — агент применит изменения сам."
