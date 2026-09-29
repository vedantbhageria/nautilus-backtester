# Open the dashboard in the browser once uvicorn is actually listening, so the
# first page load doesn't hit "connection refused".
param([int]$Port = 8000, [int]$TimeoutSec = 60)
for ($i = 0; $i -lt $TimeoutSec * 2; $i++) {
    try {
        $c = New-Object System.Net.Sockets.TcpClient
        $c.Connect("127.0.0.1", $Port)
        $c.Close()
        Start-Process "http://localhost:$Port"
        exit 0
    } catch {
        Start-Sleep -Milliseconds 500
    }
}
exit 1
