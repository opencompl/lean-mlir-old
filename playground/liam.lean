Liam O'Connor:
notation `label ` binders `, ` r:(scoped P, Asm.declareLabel P) := r
notation `for ` binders `, ` r:(scoped P, Asm.forany P) := r
notation `::`  := AsmBlock.jump
notation `next`  := Jump.continue
notation cmd `: `:25 block `;` more := Asm.seq (toAsm cmd block) more
notation `assert` `{` assn `}` rest := SyntaxBlock.code assn rest
notation `routine` rest := SyntaxBlock.sub rest
notation `given ` binders `, ` r:(scoped P, Subroutine.forany P) := r
notation `requires` `{` pre `}` `ensures` `{` post `}` `start:` block `;` rest :=
  Subroutine.post post (Subroutine.body (Asm.declareLabel (λ start, Asm.seq (Asm.block start pre block) rest)))

--instance foo  : has_andthen Asm' Asm' Asm' := { andthen := Asm.seq}
notation `bvc ` arg `;` rest := Jump.bvc arg rest
notation `bvs ` arg `;` rest := Jump.bvs arg rest
notation `bcc ` arg `;` rest := Jump.bcc arg rest
notation `bcs ` arg `;` rest := Jump.bcs arg rest
notation `bpl ` arg `;` rest := Jump.bpl arg rest
notation `bmi ` arg `;` rest := Jump.bmi arg rest
notation `bne ` arg `;` rest := Jump.bne arg rest
notation `beq ` arg `;` rest := Jump.beq arg rest
notation `jmp ` arg  := Jump.jmp arg
notation `bit ` arg `;` rest := AsmBlock.inst (Inst.bit arg) rest
notation `cmp ` arg `;` rest := AsmBlock.inst (Inst.cmp arg) rest
notation `cpx ` arg `;` rest := AsmBlock.inst (Inst.cpx arg) rest
notation `cpy ` arg `;` rest := AsmBlock.inst (Inst.cpy arg) rest
notation `lda ` arg `;` rest := AsmBlock.inst (Inst.lda arg) rest
notation `jsr ` arg `;` rest := AsmBlock.inst (Inst.jsr arg) rest
Liam O'Connor: like that

