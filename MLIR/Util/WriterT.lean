/-
## WriterT monad transformer
-/

structure WriterT (m: Type _ -> Type _) (a: Type _) where
  val: m (a × String)

def WriterT.run (wm: WriterT m a ): m (a × String) := wm.val

instance [Functor m]: Functor (WriterT m) where
  map f w := { val := Functor.map (fun (a, log) => (f a, log)) w.val }

instance [Pure m]: Pure (WriterT m) where
  pure x := { val := pure (x, "") }

instance [Monad m]: Seq (WriterT m) where
   seq mx my :=
     { val := do
        let wx <- mx.val
        let wy <- (my ()).val
        let wb := wx.fst wy.fst
        return (wb, wx.snd ++ wy.snd) }

instance [Monad m] : SeqLeft (WriterT m) where
   seqLeft mx my :=
     { val := do
        let wx <- mx.val
        let wy <- (my ()).val
        return (wx.fst, wx.snd ++ wy.snd) }

instance [Monad m] : SeqRight (WriterT m) where
   seqRight mx my :=
     { val := do
        let wx <- mx.val
        let wy <- (my ()).val
        return (wy.fst, wx.snd  ++ wy.snd ) }

instance [Bind m] [Pure m]: Bind (WriterT m) where
  bind wma a2wmb :=
    let v := do
      let (va, loga) <- wma.val
      let wb <- (a2wmb va).val
      let (vb, logb) := wb
      return (vb, loga ++ logb)
    { val := v }

def WriterT.lift [Monad m] {α : Type u} (ma: m α): WriterT m α :=
  { val := do let a <- ma; return (a, "") }

instance [Monad m]: MonadLift m (WriterT m) where
  monadLift := WriterT.lift

instance : MonadFunctor m (WriterT m) := ⟨fun f mx => { val := f (mx.val) } ⟩

instance [Monad m] : Applicative (WriterT m) where
  pure := Pure.pure
  seqLeft := SeqLeft.seqLeft
  seqRight := SeqRight.seqRight

instance [Monad m]: Monad (WriterT m) where
  pure := Pure.pure
  bind := Bind.bind
  map  := Functor.map

def logWriterT [Monad m] (s: String): WriterT.{u} m PUnit.{u+1} :=
  { val := pure (.unit, s) }
