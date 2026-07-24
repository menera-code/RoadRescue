from cryptography import x509
from cryptography.x509.oid import NameOID
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
import datetime

# Generate a private key
private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

# Build a self-signed certificate
subject = issuer = x509.Name([
    x509.NameAttribute(NameOID.COUNTRY_NAME, "PH"),
    x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "Oriental Mindoro"),
    x509.NameAttribute(NameOID.LOCALITY_NAME, "Calapan"),
    x509.NameAttribute(NameOID.ORGANIZATION_NAME, "RESQAPP"),
    x509.NameAttribute(NameOID.COMMON_NAME, "localhost"),
])

cert = x509.CertificateBuilder().subject_name(
    subject
).issuer_name(
    issuer
).public_key(
    private_key.public_key()
).serial_number(
    x509.random_serial_number()
).not_valid_before(
    datetime.datetime.utcnow()
).not_valid_after(
    datetime.datetime.utcnow() + datetime.timedelta(days=365)
).add_extension(
    x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress("127.0.0.1"),
        x509.IPAddress("192.168.10.7"),   # Replace with your laptop's IP
    ]),
    critical=False,
).sign(private_key, hashes.SHA256())

# Write the private key and certificate to files
with open("localhost-key.pem", "wb") as f:
    f.write(private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))

with open("localhost.pem", "wb") as f:
    f.write(cert.public_bytes(serialization.Encoding.PEM))

print("✅ Certificate and key generated: localhost.pem / localhost-key.pem")