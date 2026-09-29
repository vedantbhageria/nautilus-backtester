# Make sure Redis (running inside WSL) is up and STAYS up. Exit 0 when it answers PING.
#
# WSL terminates a distro ~15s after its last wsl.exe session ends, and systemd
# services such as redis-server die with it. A trading node started from Windows
# doesn't count as a WSL session, so without a held-open session Redis vanished
# mid-run (the node lost its cache, message bus and dashboard mid-trade).
# The fix: one hidden `wsl.exe ... sleep infinity` session, tagged so it's reused.
param(
    [string]$Distro = "Ubuntu",
    [int]$Port = 6379,
    [int]$TimeoutSec = 45
)
$marker = "nautilus-redis-keepalive"

function Test-RedisPing {
    try {
        $client = New-Object System.Net.Sockets.TcpClient
        $client.ReceiveTimeout = 1000
        $client.Connect("127.0.0.1", $Port)
        $stream = $client.GetStream()
        $bytes = [Text.Encoding]::ASCII.GetBytes("PING`r`n")
        $stream.Write($bytes, 0, $bytes.Length)
        $buf = New-Object byte[] 64
        $n = $stream.Read($buf, 0, 64)
        $client.Close()
        return ([Text.Encoding]::ASCII.GetString($buf, 0, $n) -like "+PONG*")
    } catch {
        return $false
    }
}

$keep = Get-CimInstance Win32_Process -Filter "Name='wsl.exe'" | Where-Object { $_.CommandLine -like "*$marker*" }
if ($keep) {
    Write-Host "[redis] WSL keep-alive already running"
} else {
    Start-Process -WindowStyle Hidden -FilePath wsl.exe `
        -ArgumentList '-d', $Distro, '-e', 'sh', '-c', "`"exec sleep infinity # $marker`""
    Write-Host "[redis] started hidden WSL keep-alive ($Distro)"
}

$deadline = (Get-Date).AddSeconds($TimeoutSec)
$kicked = $false
while ((Get-Date) -lt $deadline) {
    if (Test-RedisPing) {
        Write-Host "[redis] ready on 127.0.0.1:$Port"
        exit 0
    }
    # systemd normally starts redis-server when the distro boots; if it hasn't
    # after ~10s, start it directly (old launch.bat behaviour).
    if (-not $kicked -and ((Get-Date) -gt $deadline.AddSeconds(-($TimeoutSec - 10)))) {
        $kicked = $true
        Write-Host "[redis] not answering yet; starting redis-server in $Distro"
        wsl.exe -d $Distro -e sh -c "redis-cli ping >/dev/null 2>&1 || redis-server --daemonize yes --bind 0.0.0.0" | Out-Null
    }
    Start-Sleep -Milliseconds 500
}
Write-Host "[redis] ERROR: no PONG from 127.0.0.1:$Port after $TimeoutSec s"
exit 1
