{
  description = "My Lean package";

  inputs.lean.url = github:leanprover/lean4-nightly/nightly-2021-12-24;
  inputs.mine = ./MLIR;
  inputs.flake-utils.url = github:numtide/flake-utils;

  outputs = { pkgs ? import <nixpkgs>, self, lean, flake-utils, mine }: flake-utils.lib.eachDefaultSystem (system:
  pkgs.dockerTools.buildImage {
  name = "hello-docker";
  config = {
	    Cmd = [ "${pkgs.hello}/bin/hello" ];
	  };
  });
}
