# Script PowerShell tự tạo SSL self-signed certificate cho AI-BE

$certDir = Join-Path $PSScriptRoot 'certs'
if (!(Test-Path $certDir)) {
    New-Item -ItemType Directory -Path $certDir | Out-Null
}

$keyPath = Join-Path $certDir 'key.pem'
$certPath = Join-Path $certDir 'cert.pem'

# Sử dụng OpenSSL để tạo key và cert
$subj = '/C=VN/ST=HoChiMinh/L=District1/O=MyCompany/OU=IT/CN=localhost'
& openssl req -x509 -nodes -days 365 -newkey rsa:2048 `
    -keyout $keyPath `
    -out $certPath `
    -subj $subj

Write-Host "Đã tạo SSL key và cert tại:`n  - $keyPath`n  - $certPath"
