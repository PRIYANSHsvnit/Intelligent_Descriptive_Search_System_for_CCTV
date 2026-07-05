{
  description = "Dev shell for the Intelligent Descriptive Search System for CCTV (Turborepo: Next.js frontend + Python CV backend)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };

        # Native libraries the uv-installed wheels (torch, opencv-python,
        # ultralytics, pillow, numpy) dlopen at runtime. Exposed via
        # LD_LIBRARY_PATH so the prebuilt manylinux wheels can find them.
        runtimeLibs = with pkgs; [
          stdenv.cc.cc.lib # libstdc++ / libgcc_s (torch, numpy)
          zlib
          glib # libgthread / libglib (opencv)
          libGL # libGL.so.1 (opencv imshow / cv2)
          glib.out
        ] ++ lib.optionals stdenv.isLinux [
          # X/GL bits pulled in by opencv even in headless setups
          xorg.libX11
          xorg.libXext
          xorg.libXrender
          xorg.libSM
          xorg.libICE
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            # JS / TS toolchain
            nodejs_22
            pnpm
            turbo

            # Python toolchain (backend managed by uv via uv.lock)
            python311
            uv
          ];

          # Let prebuilt Python wheels find their native deps.
          LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath runtimeLibs;

          # Keep uv from trying to download its own Python; use the Nix one.
          env = {
            UV_PYTHON = "${pkgs.python311}/bin/python3.11";
            UV_PYTHON_DOWNLOADS = "never";
          };

          shellHook = ''
            echo "CCTV search dev shell"
            echo "  node    $(node --version)"
            echo "  pnpm    $(pnpm --version)"
            echo "  python  $(python3 --version)"
            echo "  uv      $(uv --version)"
            echo ""
            echo "Frontend + workspace deps:  pnpm install"
            echo "Run everything (turbo):     pnpm dev"
            echo "Backend only:               cd apps/backend_py && uv sync && pnpm dev"
          '';
        };
      });
}
