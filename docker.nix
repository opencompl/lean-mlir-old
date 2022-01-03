{ pkgs ? import <nixpkgs> {} }:

with pkgs;

# give me bash, git, busybox
# and the environment of lean-mlir (ie, whatever the flake needs to build)
pkgs.dockerTools.buildImage {
  name = "hello-docker";
  config = {
    Cmd = [ "${pkgs.hello}/bin/hello" ];
  };
}
