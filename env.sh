#!/bin/bash
# SLB PoV — environment config
# Usage: source env.sh, then `tfy apply` any yaml/

export WORKSPACE_FQN="tfy-slb-demo:slb-ws"
export ML_REPO_FQN="slb-pilot"

# Artifact FQNs
export ARTIFACT_VJEPA_VITL="model:slb-pilot/slb-pov/vjepa2-vitl:1"
export ARTIFACT_VJEPA_VITH="model:slb-pilot/slb-pov/vjepa2-vith:1"
export ARTIFACT_VJEPA_VITG="model:slb-pilot/slb-pov/vjepa2-vitg:1"
export ARTIFACT_VJEPA_VITG_384="model:slb-pilot/slb-pov/vjepa2-vitg-384:1"
export ARTIFACT_SFM_BASE="model:slb-pilot/slb-pov/sfm-base:1"
export ARTIFACT_SFM_BASE_512="model:slb-pilot/slb-pov/sfm-base-512:1"
export ARTIFACT_SFM_LARGE="model:slb-pilot/slb-pov/sfm-large:1"
export ARTIFACT_SFM_LARGE_512="model:slb-pilot/slb-pov/sfm-large-512:1"
export ARTIFACT_SFM_DATA="artifact:slb-pilot/slb-pov/sfm-facies-data:1"

# Internal encoder URL (derived from workspace name)
export ENCODER_URL_VITL="http://vjepa2-vitl.${WORKSPACE_FQN##*:}.svc.cluster.local:8000"
