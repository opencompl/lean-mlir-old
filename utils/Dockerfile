FROM nixpkgs/nix 
RUN git clone  https://github.com/opencompl/lean-mlir
RUN cd lean-mlir && nix-shell https://github.com/leanprover/lean4/archive/master.tar.gz -A nix --command "nix-build" 
RUN cd lean-mlir && nix-shell https://github.com/leanprover/lean4/archive/master.tar.gz -A nix --command "nix-build .#executable" 
