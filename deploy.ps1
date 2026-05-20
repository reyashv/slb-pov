# deploy.ps1
# Substitutes env vars in yamls and deploys all services and jobs to TrueFoundry
# Usage: .\deploy.ps1

# ── Config ────────────────────────────────────────────
$WORKSPACE_FQN       = "tfy-slb-demo:slb-ws"
$ENCODER_URL_VITL    = "http://vjepa2-vitl.slb-ws.svc.cluster.local:8000"
$ARTIFACT_VJEPA_VITL = "model:slb-pilot/slb-pov/vjepa2-vitl:1"
$ARTIFACT_VJEPA_VITH = "model:slb-pilot/slb-pov/vjepa2-vith:1"
$ARTIFACT_VJEPA_VITG = "model:slb-pilot/slb-pov/vjepa2-vitg:1"
$ARTIFACT_VJEPA_VITG_384 = "model:slb-pilot/slb-pov/vjepa2-vitg-384:1"
$ARTIFACT_SFM_BASE       = "model:slb-pilot/slb-pov/sfm-base:1"
$ARTIFACT_SFM_BASE_512   = "model:slb-pilot/slb-pov/sfm-base-512:1"
$ARTIFACT_SFM_LARGE      = "model:slb-pilot/slb-pov/sfm-large:1"
$ARTIFACT_SFM_LARGE_512  = "model:slb-pilot/slb-pov/sfm-large-512:1"
$ARTIFACT_SFM_DATA       = "artifact:slb-pilot/slb-pov/sfm-facies-data:1"

# ── Temp dir for substituted yamls ───────────────────
$TMP = "$env:TEMP\tfy-deploy"
New-Item -ItemType Directory -Force -Path $TMP | Out-Null

function Apply-Yaml($file) {
    $name = Split-Path $file -Leaf
    $content = Get-Content $file -Raw

    # Substitute all variables
    $content = $content -replace '\$\{WORKSPACE_FQN\}',       $WORKSPACE_FQN
    $content = $content -replace '\$\{ENCODER_URL_VITL\}',    $ENCODER_URL_VITL
    $content = $content -replace '\$\{ARTIFACT_VJEPA_VITL\}', $ARTIFACT_VJEPA_VITL
    $content = $content -replace '\$\{ARTIFACT_VJEPA_VITH\}', $ARTIFACT_VJEPA_VITH
    $content = $content -replace '\$\{ARTIFACT_VJEPA_VITG\}', $ARTIFACT_VJEPA_VITG
    $content = $content -replace '\$\{ARTIFACT_VJEPA_VITG_384\}', $ARTIFACT_VJEPA_VITG_384
    $content = $content -replace '\$\{ARTIFACT_SFM_BASE\}',     $ARTIFACT_SFM_BASE
    $content = $content -replace '\$\{ARTIFACT_SFM_BASE_512\}', $ARTIFACT_SFM_BASE_512
    $content = $content -replace '\$\{ARTIFACT_SFM_LARGE\}',    $ARTIFACT_SFM_LARGE
    $content = $content -replace '\$\{ARTIFACT_SFM_LARGE_512\}',$ARTIFACT_SFM_LARGE_512
    $content = $content -replace '\$\{ARTIFACT_SFM_DATA\}',     $ARTIFACT_SFM_DATA

    # Write substituted yaml to temp file
    $tmp_file = "$TMP\$name"
    $content | Set-Content $tmp_file -Encoding UTF8

    Write-Host "Deploying $name..." -ForegroundColor Cyan
    tfy apply -f $tmp_file
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR deploying $name" -ForegroundColor Red
    } else {
        Write-Host "OK: $name" -ForegroundColor Green
    }
}

# ── Deploy order: encoders first, then decoders, then jobs ──

Write-Host "`n=== SFM Encoders ===" -ForegroundColor Yellow
#Apply-Yaml "yaml\sfm-base.yaml"
Apply-Yaml "yaml\sfm-base-512.yaml"
Apply-Yaml "yaml\sfm-large.yaml"
Apply-Yaml "yaml\sfm-large-512.yaml"

Write-Host "`n=== V-JEPA Encoders ===" -ForegroundColor Yellow
Apply-Yaml "yaml\vjepa2-vitl.yaml"
Apply-Yaml "yaml\vjepa2-vith.yaml"
Apply-Yaml "yaml\vjepa2-vitg-1.yaml"
Apply-Yaml "yaml\vjepa2-vitg-384-1.yaml"

Write-Host "`n=== Decoders (decoupled) ===" -ForegroundColor Yellow
Apply-Yaml "yaml\decoder-classify.yaml"
Apply-Yaml "yaml\decoder-detect.yaml"
Apply-Yaml "yaml\decoder-segment.yaml"

Write-Host "`n=== Fused Services ===" -ForegroundColor Yellow
Apply-Yaml "yaml\fused-classify.yaml"
Apply-Yaml "yaml\fused-detect.yaml"
Apply-Yaml "yaml\fused-segment.yaml"

Write-Host "`n=== Jobs ===" -ForegroundColor Yellow
Apply-Yaml "yaml\sfm-finetune.yaml"
Apply-Yaml "yaml\vjepa-probe-classify.yaml"
Apply-Yaml "yaml\vjepa-probe-segment.yaml"
Apply-Yaml "yaml\vjepa-joint-finetune.yaml"

Write-Host "`nAll done!" -ForegroundColor Green
