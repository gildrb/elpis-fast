# Compatibility import and option namespace; Docker owns runtime lifecycle.
{ config, lib, pkgs, ... }:
let
  cfg = config.workstation.qwenInference;
  compose = import ./compose.nix { inherit pkgs; };
  command = "${pkgs.docker-compose}/bin/docker-compose --project-name qwen-inference --file ${compose}";
in {
  options.workstation.qwenInference = {
    enable = lib.mkEnableOption "the canonical Qwen EXL3 Docker deployment";
    stateRoot = lib.mkOption {
      type = lib.types.externalPath;
      default = "/srv/ai/models/qwen3.8-27b";
      description = "Existing private state, models, key, cache, prefix-cache and lock. Host-provisioned.";
    };
    requiredMountPoint = lib.mkOption {
      type = lib.types.nullOr lib.types.externalPath;
      default = null;
      description = "Optional host mount checked before Compose startup.";
    };
    port = lib.mkOption { type = lib.types.port; default = 18020; };
    model = lib.mkOption {
      type = lib.types.str;
      default = "qwen3.8-27b";
      readOnly = true;
    };
    allowUnqualified = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Explicitly permit the unqualified EXL3 native-context candidate.";
    };
    image = lib.mkOption {
      type = lib.types.str;
      default = "qwen-inference:exl3";
      description = "Prebuilt reviewed Docker image; use its immutable digest for deployment.";
    };
  };
  config = lib.mkIf cfg.enable {
    systemd.user.services.qwen-inference = {
      description = "Qwen EXL3 canonical Docker deployment";
      wantedBy = [ "default.target" ];
      requires = [ "docker.service" ];
      after = [ "docker.service" ];
      serviceConfig = {
        Type = "oneshot";
        RemainAfterExit = true;
        Environment = [
          "DOCKER_HOST=unix://%t/docker.sock"
          "QWEN_STATE_ROOT=${cfg.stateRoot}"
          "QWEN_PORT=${toString cfg.port}"
          "QWEN_IMAGE=${cfg.image}"
          "QWEN_ALLOW_UNQUALIFIED=${if cfg.allowUnqualified then "1" else "0"}"
        ];
        ExecStartPre = lib.optional (cfg.requiredMountPoint != null)
          "${pkgs.util-linux}/bin/mountpoint --quiet -- ${lib.escapeShellArg cfg.requiredMountPoint}";
        ExecStart = "${command} up --detach --no-build --pull never --wait --wait-timeout 1200";
        ExecStop = "${command} stop --timeout 60";
        TimeoutStartSec = "25min";
        TimeoutStopSec = "90s";
        UMask = "0077";
      };
    };
  };
}
