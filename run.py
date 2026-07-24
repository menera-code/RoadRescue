import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        ssl_keyfile="localhost-key.pem",
        ssl_certfile="localhost.pem",
        reload=True,   # optional, for development
    )