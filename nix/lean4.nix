{ pkgs }:
# Official Lean 4.34.0 release: Bend 2.0.35's `--verdict` builds its BendTT kernel with exactly this
# version (nixpkgs' lean4 lags it). Bump together with nix/bend.nix.
pkgs.stdenv.mkDerivation (finalAttrs: {
  pname = "lean4-bin";
  version = "4.34.0";

  src = pkgs.fetchurl {
    url = "https://github.com/leanprover/lean4/releases/download/v${finalAttrs.version}/lean-${finalAttrs.version}-linux.tar.zst";
    hash = "sha256-yqqYNWCYyF3A/LvSjh7GbznrZVGCmXK3Uv8g4ShrZGs=";
  };

  nativeBuildInputs = [
    pkgs.autoPatchelfHook
    pkgs.zstd
  ];
  buildInputs = [ pkgs.stdenv.cc.cc.lib ];

  dontConfigure = true;
  dontBuild = true;
  dontStrip = true;

  installPhase = ''
    runHook preInstall
    mkdir -p "$out"
    cp -R bin include lib share src LICENSE LICENSES "$out/"
    runHook postInstall
  '';

  preFixup = ''
    addAutoPatchelfSearchPath "$out/lib" "$out/lib/lean"
  '';

  meta = {
    description = "Lean 4 theorem prover (official release build)";
    homepage = "https://lean-lang.org";
    license = pkgs.lib.licenses.asl20;
    mainProgram = "lean";
    platforms = [ "x86_64-linux" ];
  };
})
