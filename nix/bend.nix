{ pkgs }:

let
  buildHelper = ../bend/build_toolchain.py;
  source = pkgs.fetchurl {
    url = "https://codeload.github.com/bendlang/bend/tar.gz/79df8d9c40722ee9507a1e253f283b51025f9d6c";
    sha256 = "ad7ac21c0145dacaab012ff6fc6c359e57f956cd6940cf5bd94a1b8b12745949";
  };
in
pkgs.stdenv.mkDerivation {
  pname = "bend";
  version = "2.0.35";

  src = pkgs.fetchurl {
    url = "https://github.com/bendlang/bend/releases/download/v2.0.35/bend-2.0.35-linux-x64.tar.gz";
    sha256 = "63039d1a119f716767ac5a7d8fe0717cfacf219c6c253c35192148e0dade722f";
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
    description = "Unmodified Bend 2.0.35 release checker/compiler and Base";
    homepage = "https://github.com/bendlang/bend";
    license = pkgs.lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
