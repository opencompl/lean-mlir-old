{
  description = "My Lean package";

  # inputs.nixpkgs.url = github:NixOS/nixpkgs/nixos-21.11;
  inputs.nixpkgs.url = github:NixOS/nixpkgs/21.11;
  inputs.lean.url = github:leanprover/lean4-nightly/nightly-2021-12-24;
  inputs.flake-utils.url = github:numtide/flake-utils;

  outputs = { self, lean, flake-utils, nixpkgs, ... }: flake-utils.lib.eachDefaultSystem (system:
  let
      pkgs = nixpkgs.legacyPackages.${system};
      leanPkgs = lean.packages.${system};
      pkg = leanPkgs.buildLeanPackage {
        name = "MLIR";  # must match the name of the top-level .lean file
        src = ./.;
      };
    in {
      packages = pkg // {
        inherit (leanPkgs) lean;
      };
      defaultPackage = pkg.modRoot;
    });
}
