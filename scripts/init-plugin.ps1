#!/usr/bin/env pwsh
$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$scriptDir = Split-Path -Path $MyInvocation.MyCommand.Path -Parent
if (-not $scriptDir) {
    $scriptDir = Get-Location
}

$pythonScript = Join-Path $scriptDir 'init-plugin.py'

& python $pythonScript @args