{ pkgs }:

let
  buildHelper = ../bend/build_toolchain.py;
  source = pkgs.fetchurl {
    url = "https://codeload.github.com/bendlang/bend/tar.gz/7d8a3eb036042c6549461054d25a10f26d361c5c";
    sha256 = "2232ba4b9f66f5d3729e47417994952b484e8b5f44823a682c08bf68c51eaa00";
  };
in
pkgs.stdenv.mkDerivation {
  pname = "bend";
  version = "2.0.34";

  src = pkgs.fetchurl {
    url = "https://github.com/bendlang/bend/releases/download/v2.0.34/bend-2.0.34-linux-x64.tar.gz";
    sha256 = "78106a97af242429dcc057258eb8d10f69cddebcd5e263022185a52d003e09bf";
  };
  inherit source;
  dontUnpack = true;
  nativeBuildInputs = [
    pkgs.makeWrapper
    pkgs.autoPatchelfHook
    pkgs.python3
  ];
  buildInputs = [ pkgs.stdenv.cc.cc ];
  dontConfigure = true;
  # Match the selected package's working ELF relocation; never rebuild its bundle.
  dontStrip = true;
  # The release launcher is hash-pinned byte-for-byte; /bin/sh exists at runtime.
  dontPatchShebangs = true;

  buildPhase = ''
    runHook preBuild
    python3 ${buildHelper} build \
      "$src" "$source" \
      staged toolchain.json --nix-relocation
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p "$out/bin" "$out/libexec/bend" "$out/share/bend"
    cp -r staged/bin staged/bend2 staged/guide "$out/libexec/bend/"
    makeWrapper "$out/libexec/bend/bin/bend" "$out/bin/bend" \
      --prefix PATH : "${pkgs.lib.makeBinPath [ pkgs.llvmPackages_19.clang ]}" \
      --set BEND_NO_TELEMETRY 1
    runHook postInstall
  '';

  postPhases = [ "recordToolchainPhase" ];
  recordToolchainPhase = ''
    python3 ${buildHelper} record-relocated \
      "$src" "$source" "$out/libexec/bend" \
      "$out/share/bend/toolchain.json" ${./bend.nix}
  '';


  meta = {
    description = "Unmodified Bend 2.0.34 release checker/compiler and Base";
    homepage = "https://github.com/bendlang/bend";
    license = pkgs.lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
