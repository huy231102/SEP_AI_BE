#!/usr/bin/env bash
# Script tự tạo chứng chỉ self-signed cho AI-BE

# Tạo thư mục certs nếu chưa tồn tại
dir="$(dirname "$0")/certs"
mkdir -p "$dir"

# Đường dẫn lưu key và cert
KEY_PATH="$dir/key.pem"
CERT_PATH="$dir/cert.pem"

# Tạo key và chứng chỉ self-signed
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout "$KEY_PATH" \
  -out "$CERT_PATH" \
  -subj "/C=VN/ST=HoChiMinh/L=District1/O=MyCompany/OU=IT/CN=localhost"

echo "Đã tạo SSL key và cert tại:"
echo "  - $KEY_PATH"
echo "  - $CERT_PATH"
