<#
.SYNOPSIS
  Обновление агента photo-restore до сборки, которая работает на сервере.

.DESCRIPTION
  Узнаёт номер сборки у сервера (/healthz, адрес — из config.toml), скачивает с GitHub исходники
  этого тега и ставит из них агента в agent\.venv. На время установки задача Планировщика
  останавливается, потом запускается снова. config.toml не трогает; рабочая копия репозитория
  в установке не участвует.

    powershell -ExecutionPolicy Bypass -File agent\update.ps1
    powershell -ExecutionPolicy Bypass -File agent\update.ps1 -Version 0.2.1

  Первая установка — agent\install.ps1.
#>
param(
    [string]$Version = "",
    [string]$TaskName = "photo-restore agent"
)
$ErrorActionPreference = "Stop"

$Python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Config = Join-Path $PSScriptRoot "config.toml"
if (-not (Test-Path $Python)) { throw "Агент не установлен: сначала agent\install.ps1" }

$PyArgs = @("-m", "restore_agent.update", "--config", $Config, "--task", $TaskName)
if ($Version) { $PyArgs += @("--version", $Version) }
& $Python @PyArgs
exit $LASTEXITCODE
