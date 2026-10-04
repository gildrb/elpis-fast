{
  description = "Qwen3.8-27B Docker recipe and pinned development tools";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/d482ef84049d9b7276b83a06e4e4d76983830097";

  outputs = { self, nixpkgs }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };
      standaloneCompose = import ./nix/compose.nix { inherit pkgs; };
      bend = import ./nix/bend.nix { inherit pkgs; };
      lean4 = import ./nix/lean4.nix { inherit pkgs; };
      # The pinned bend with Lean 4.34.0 on PATH, for `bend PROOF.bend --verdict` (BendTT kernel
      # build). A separate output, so `bend`'s store path, which the acceptor identity records,
      # stays unchanged.
      bend-verdict = pkgs.writeShellApplication {
        name = "bend";
        runtimeInputs = [ lean4 ];
        text = ''exec ${bend}/bin/bend "$@"'';
      };
      serve = pkgs.writeShellApplication {
        name = "qwen-serve";
        runtimeInputs = [ pkgs.docker-compose ];
        text = ''
          if [[ "''${1:-}" == "--state-root" && $# == 2 ]]; then
            export QWEN_STATE_ROOT="$2"
          elif (( $# > 0 )); then
            echo "Usage: qwen-serve --state-root /absolute/existing/state" >&2
            exit 1
          fi
          : "''${QWEN_STATE_ROOT:?Set an existing absolute state directory}"
          exec docker-compose --project-name qwen-inference --file ${standaloneCompose} up --no-build --pull never
        '';
      };
      reference = nixpkgs.lib.nixosSystem {
        inherit system;
        specialArgs.username = "qwen";
        modules = [
          self.nixosModules.reference
          {
            # Evaluation fixture only: not a bootable machine configuration.
            system.stateVersion = "26.05";
            nixpkgs.config.allowUnfree = true;
          }
        ];
      };
    in {
      nixosModules.default = self.nixosModules.qwen-inference;
      nixosModules.qwen-inference = import ./nix/qwen-inference.nix;
      nixosModules.reference = import ./nix/reference.nix;

      packages.${system} = { deployment = standaloneCompose; inherit serve bend bend-verdict; };
      apps.${system}.serve = {
        type = "app";
        meta.description = "Run the canonical Docker deployment in the foreground";
        program = "${serve}/bin/qwen-serve";
      };

      devShells.${system}.default = pkgs.mkShellNoCC {
        LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc ];
        packages = with pkgs; [
          bend
          llvmPackages_19.clang
          python313
          python312
          git
          git-lfs
          uv
          docker-client
          docker-compose
          curl
          jq
          shellcheck
          util-linux
        ];
      };

      # Evaluate the thin adapter only. No daemon, models or GPU are needed.
      checks.${system} = {
      standalone-launcher = serve;
      serving-units = pkgs.linkFarm "qwen-serving-units" [
        {
          name = "qwen-inference.service";
          path = reference.config.systemd.user.units."qwen-inference.service".unit;
        }
      ];
      };
    };
}
