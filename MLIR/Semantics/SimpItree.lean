/- Simplification tactic simp_itree -/

import Lean

-- See: https://leanprover.zulipchat.com/#narrow/stream/270676-lean4/topic/How.20to.20use.20.60registerSimpAttr.60.3F/near/275925106

open Lean Meta in
initialize SimpItreeExtension:
  SimpExtension ← registerSimpAttr `simp_itree "ITree simp set"
