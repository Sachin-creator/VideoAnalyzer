param(
    [string]$Out = "out_sync_test_safe.mkv",
    [int]$Duration = 5,
    [int]$Fps = 25,
    [int]$Freq = 1000
)

# Safe generator for PowerShell: write filter to a temporary file and call ffmpeg with -filter_complex_script
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition
$tmp = [System.IO.Path]::GetTempPath()
$filterFile = [System.IO.Path]::Combine($tmp, [System.IO.Path]::GetRandomFileName() + '.filt')

[System.IO.File]::WriteAllText($filterFile, "[2:v]format=rgba[flash];`n[0:v][flash]overlay=enable=lt(mod(t\,1)\,1/$Fps)[v];`n[1:a]volume=if(lt(mod(t\,1)\,1/$Fps)\,20\,0)[a]")
Write-Host "Wrote filter file: $filterFile"

$ffmpeg = "ffmpeg"
$args = @(
    '-y',
    '-f','lavfi','-i',"color=black:s=1280x720:rate=$Fps:d=$Duration",
    '-f','lavfi','-i',"sine=frequency=$Freq:sample_rate=44100:duration=$Duration",
    '-f','lavfi','-i',"color=white:s=1280x720:rate=$Fps:d=$Duration",
    '-filter_complex_script', $filterFile,
    '-map','[v]','-map','[a]','-r',$Fps.ToString(),'-c:v','libx264','-pix_fmt','yuv420p','-c:a','pcm_s16le',$Out
)

& $ffmpeg @args

$wav = [System.IO.Path]::ChangeExtension($Out, '.wav')
$png = [System.IO.Path]::ChangeExtension($Out, '_waveform.png')
& $ffmpeg -y -i $Out -vn -ac 1 -ar 44100 $wav
& $ffmpeg -y -i $wav -lavfi "aformat=channel_layouts=mono,showwavespic=s=1280x256:colors=white" -frames:v 1 $png

Write-Host "Created: $Out, $wav, $png"
[System.IO.File]::Delete($filterFile)
