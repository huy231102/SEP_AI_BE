// Node script tự tạo SSL self-signed certs bằng package 'selfsigned'
const selfsigned = require('selfsigned');
const fs = require('fs');
const path = require('path');

// Thư mục lưu certs
const certDir = path.join(__dirname, 'certs');
if (!fs.existsSync(certDir)) fs.mkdirSync(certDir);

// Tạo chứng chỉ
const attrs = [{ name: 'commonName', value: '192.168.0.104' }];
// Tạo chứng chỉ với subjectAltName gồm localhost và địa chỉ IP của máy (192.168.0.104)
const pems = selfsigned.generate(attrs, {
  days: 365,
  keySize: 4096,
  algorithm: 'sha256',
  extensions: [{ name: 'basicConstraints', cA: true }],
  altNames: [
    { type: 2, value: 'localhost' }, // DNS
    { type: 7, ip: '127.0.0.1' },    // IPv4 localhost
    { type: 7, ip: '192.168.0.104' } // IPv4 local network
  ]
});

// Ghi file
fs.writeFileSync(path.join(certDir, 'key.pem'), pems.private);
fs.writeFileSync(path.join(certDir, 'cert.pem'), pems.cert);

console.log('Đã tạo SSL key và cert tại:');
console.log(' -', path.join(certDir, 'key.pem'));
console.log(' -', path.join(certDir, 'cert.pem'));
