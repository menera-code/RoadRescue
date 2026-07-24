from security import create_access_token, decode_token

token = create_access_token(sub="1", role="admin")
print("TOKEN:", token)

payload = decode_token(token)
print("PAYLOAD:", payload)
