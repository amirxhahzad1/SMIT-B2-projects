# One-click authentication helper for Gmail and Google Calendar MCP
param(
    [ValidateSet("all", "gmail", "calendar")]
    [string]$Service = "all"
)

$CredentialsPath = Join-Path $PSScriptRoot "..\gcp-oauth.keys.json"
if (-not (Test-Path $CredentialsPath)) {
    Write-Host "Error: gcp-oauth.keys.json not found in project root!" -ForegroundColor Red
    exit 1
}

$CredentialsResolved = (Resolve-Path $CredentialsPath).Path

if ($Service -in @("all", "gmail")) {
    Write-Host "`n=== Authorizing Gmail MCP ===" -ForegroundColor Cyan
    Write-Host "Starting Gmail auth server. A browser window will open..." -ForegroundColor Yellow
    npx -y @gongrzhe/server-gmail-autoauth-mcp auth
}

if ($Service -in @("all", "calendar")) {
    Write-Host "`n=== Authorizing Google Calendar MCP ===" -ForegroundColor Cyan
    $env:GOOGLE_OAUTH_CREDENTIALS = $CredentialsResolved
    $AuthServer = Join-Path $env:APPDATA "npm\node_modules\@cocal\google-calendar-mcp\build\auth-server.js"
    if (Test-Path $AuthServer) {
        Write-Host "Starting Calendar auth server. A browser window will open..." -ForegroundColor Yellow
        node $AuthServer
    } else {
        Write-Host "Running calendar auth via npx..." -ForegroundColor Yellow
        npx -y @cocal/google-calendar-mcp auth
    }
}

Write-Host "`nDone! Run 'python main.py' to launch your assistant." -ForegroundColor Green
