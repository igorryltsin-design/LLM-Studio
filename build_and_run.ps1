[CmdletBinding()]
param(
    [switch]$Dev,
    [switch]$Preview,
    [switch]$Install,
    [switch]$SkipBase,
    [string]$PythonPath,
    [string]$BaseHost = '127.0.0.1',
    [int]$BasePort = 8001
)

$ErrorActionPreference = 'Stop'

function Resolve-ProjectRoot {
    $scriptPath = $MyInvocation.MyCommand.Path
    if (-not $scriptPath) {
        throw "Не удалось определить расположение скрипта."
    }
    return (Split-Path -Parent $scriptPath)
}

function Resolve-PythonExe {
    param([string]$Preferred)

    if ($Preferred) {
        return $Preferred
    }

    if ($env:PYTHON) {
        return $env:PYTHON
    }

    foreach ($candidate in @('python', 'python3')) {
        $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
        if ($cmd) {
            return $cmd.Source
        }
    }

    throw "Интерпретатор Python не найден. Укажите путь через параметр -PythonPath."
}

function Require-Command {
    param(
        [string]$Name,
        [string]$Message
    )

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw $Message
    }
}

function Invoke-ExternalCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string[]]$Arguments = @(),
        [string]$Description,
        [long[]]$AcceptableExitCodes = @(0)
    )

    & $Command @Arguments
    $exitCode = $LASTEXITCODE

    if ($null -eq $exitCode) {
        return
    }

    if (-not ($AcceptableExitCodes -contains [long]$exitCode)) {
        $label = if ([string]::IsNullOrWhiteSpace($Description)) {
            (@($Command) + $Arguments) -join ' '
        } else {
            $Description
        }

        throw "$label завершился с ошибкой (код $exitCode)."
    }
}

function Test-PythonDependencies {
    param(
        [string]$PythonExe,
        [string]$VendorPath
    )

    $code = @"
import importlib.util
import sys
modules = ("fastapi", "uvicorn", "transformers", "torch", "psutil", "sentencepiece", "accelerate", "peft")
missing = [m for m in modules if importlib.util.find_spec(m) is None]
if missing:
    print(','.join(missing))
    sys.exit(1)
"@

    $previousPath = $env:PYTHONPATH
    $previousNoSite = $env:PYTHONNOUSERSITE

    try {
        $env:PYTHONPATH = $VendorPath
        $env:PYTHONNOUSERSITE = '1'
        $output = & $PythonExe '-S' '-c' $code 2>&1
        $success = $LASTEXITCODE -eq 0
        $message = ($output | Out-String).Trim()
        return [pscustomobject]@{
            Success = $success
            Message = $message
        }
    }
    finally {
        if ($null -eq $previousPath) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $previousPath
        }

        if ($null -eq $previousNoSite) {
            Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONNOUSERSITE = $previousNoSite
        }
    }
}

$projectRoot = Resolve-ProjectRoot
Push-Location $projectRoot

try {
    $mode = 'preview'
    if ($Dev) { $mode = 'dev' }
    if ($Preview) { $mode = 'preview' }
    if ($Dev -and $Preview) {
        throw 'Нельзя одновременно указать параметры -Dev и -Preview.'
    }

    Write-Host "[Инфо] Каталог проекта: $projectRoot"

    Require-Command -Name 'npm' -Message 'npm не найден. Установите Node.js и npm либо добавьте их в PATH.'
    $pythonExe = Resolve-PythonExe -Preferred $PythonPath
    Write-Host "[Инфо] Использую Python: $pythonExe"

    $pythonLibPath = Join-Path $projectRoot 'python_libs'
    $nodeModulesPath = Join-Path $projectRoot 'node_modules'
    $requirementsPath = Join-Path $projectRoot 'requirements.txt'

    if ($Install -or -not (Test-Path $nodeModulesPath)) {
        Write-Host '[Инфо] Устанавливаю зависимости npm...'
        Invoke-ExternalCommand -Command 'npm' -Arguments @('install') -Description 'npm install'
    }

    $dependencyCheck = $null
    if (-not $SkipBase) {
        if (-not (Test-Path $pythonLibPath)) {
            New-Item -ItemType Directory -Path $pythonLibPath -ErrorAction SilentlyContinue | Out-Null
        }

        if ($Install) {
            Write-Host '[Инфо] Устанавливаю зависимости Python в каталоге python_libs...'
            Invoke-ExternalCommand -Command $pythonExe -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip') -Description 'pip install --upgrade pip'
            Invoke-ExternalCommand -Command $pythonExe -Arguments @('-m', 'pip', 'install', '--upgrade', '--target', $pythonLibPath, '-r', $requirementsPath) -Description 'pip install зависимостей'
        }

        $dependencyCheck = Test-PythonDependencies -PythonExe $pythonExe -VendorPath $pythonLibPath
        if (-not $dependencyCheck.Success) {
            $missingInfo = if ([string]::IsNullOrWhiteSpace($dependencyCheck.Message)) { 'не определено' } else { $dependencyCheck.Message }
            Write-Host "[Инфо] Требуется установка/обновление Python-зависимостей (отсутствуют: $missingInfo)"
            Write-Host '[Инфо] Устанавливаю зависимости Python в каталоге python_libs...'
            Invoke-ExternalCommand -Command $pythonExe -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip') -Description 'pip install --upgrade pip'
            Invoke-ExternalCommand -Command $pythonExe -Arguments @('-m', 'pip', 'install', '--upgrade', '--target', $pythonLibPath, '-r', $requirementsPath) -Description 'pip install зависимостей'

            $dependencyCheck = Test-PythonDependencies -PythonExe $pythonExe -VendorPath $pythonLibPath
            if (-not $dependencyCheck.Success) {
                throw "Не удалось подготовить Python-зависимости: $($dependencyCheck.Message)"
            }
        }
    }

    if ($mode -ne 'dev') {
        Write-Host '[Инфо] Собираю production-бандл фронтенда...'
        Invoke-ExternalCommand -Command 'npm' -Arguments @('run', 'build') -Description 'npm run build'
    } else {
        Write-Host '[Инфо] Запуск в dev-режиме — сборка пропущена.'
    }

    $baseProcess = $null
    $previousPythonPath = $env:PYTHONPATH
    $previousNoSite = $env:PYTHONNOUSERSITE

    try {
        if (-not $SkipBase) {
            Write-Host "[Инфо] Запускаю сервер базовой модели на http://$BaseHost:$BasePort"
            $env:PYTHONPATH = if ($previousPythonPath) {
                "$pythonLibPath$([System.IO.Path]::PathSeparator)$previousPythonPath"
            } else {
                $pythonLibPath
            }
            $env:PYTHONNOUSERSITE = '1'

            $baseProcess = Start-Process -FilePath $pythonExe -ArgumentList @('-m', 'uvicorn', 'base_model_server:app', '--host', $BaseHost, '--port', $BasePort) -WorkingDirectory $projectRoot -PassThru
            Start-Sleep -Seconds 2
            if ($baseProcess.HasExited) {
                throw 'Не удалось запустить сервер базовой модели. Проверьте вывод ошибок выше.'
            }
        }

        if ($mode -eq 'dev') {
            Write-Host '[Инфо] Запускаю фронтенд Vite в dev-режиме...'
            Invoke-ExternalCommand -Command 'npm' -Arguments @('run', 'dev') -Description 'npm run dev' -AcceptableExitCodes @(0, 130, -1073741510, 3221225786)
        } else {
            Write-Host '[Инфо] Запускаю фронтенд Vite в режиме предпросмотра...'
            Invoke-ExternalCommand -Command 'npm' -Arguments @('run', 'preview', '--', '--host') -Description 'npm run preview' -AcceptableExitCodes @(0, 130, -1073741510, 3221225786)
        }
    }
    finally {
        if ($baseProcess -and -not $baseProcess.HasExited) {
            Write-Host '[Инфо] Останавливаю сервер базовой модели...'
            try {
                Stop-Process -Id $baseProcess.Id -Force -ErrorAction SilentlyContinue
                Wait-Process -Id $baseProcess.Id -ErrorAction SilentlyContinue
            } catch {
                Write-Warning "Не удалось корректно остановить сервер базовой модели: $_"
            }
        }

        if ($null -eq $previousPythonPath) {
            Remove-Item Env:PYTHONPATH -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONPATH = $previousPythonPath
        }

        if ($null -eq $previousNoSite) {
            Remove-Item Env:PYTHONNOUSERSITE -ErrorAction SilentlyContinue
        } else {
            $env:PYTHONNOUSERSITE = $previousNoSite
        }
    }
}
finally {
    Pop-Location
}
