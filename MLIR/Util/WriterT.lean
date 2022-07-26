/-
## WriterT monad transformer
-/

def WriterT (m: Type _ -> Type _) (a: Type _) := m (a × String)

def WriterT.run (wm: WriterT m a ): m (a × String) := wm

def WriterT.mk (x: m (a × String)): WriterT m a := x

instance [Functor m]: Functor (WriterT m) where
  map f w := Functor.map (f := m) (fun (a, log) => (f a, log)) w

instance [Pure m]: Pure (WriterT m) where
  pure x := pure (f := m) (x, "")

instance [Monad m]: Seq (WriterT m) where
   seq mx my := WriterT.mk do
    let wx <- mx
    let wy <- (my ())
    let wb := wx.fst wy.fst
    return (wb, wx.snd ++ wy.snd)

instance [Monad m] : SeqLeft (WriterT m) where
   seqLeft mx my := WriterT.mk do
    let wx <- mx
    let wy <- (my ())
    return (wx.fst, wx.snd ++ wy.snd)

instance [Monad m] : SeqRight (WriterT m) where
   seqRight mx my := WriterT.mk do
    let wx <- mx
    let wy <- (my ())
    return (wy.fst, wx.snd  ++ wy.snd )

def WriterT.bindCont [Bind m] [Pure m] (k: α → WriterT m β) (x: α × String):
    WriterT m β := WriterT.mk do
  let y ← k x.fst
  return (y.fst, x.snd ++ y.snd)

def WriterT.bind [Bind m] [Pure m] (wma: WriterT m α) (a2wmb: α → WriterT m β):
    WriterT m β :=
  WriterT.mk do
    let x <- wma
    WriterT.bindCont a2wmb x

instance [Bind m] [Pure m]: Bind (WriterT m) where
  bind wma a2wmb := WriterT.bind wma a2wmb

def WriterT.lift [Monad m] {α : Type u} (ma: m α): WriterT m α :=
  Bind.bind (m := m) ma (fun a => return (a, ""))

instance [Monad m]: MonadLift m (WriterT m) where
  monadLift := WriterT.lift

instance : MonadFunctor m (WriterT m) where
  monadMap f := f

instance [Monad m] : Applicative (WriterT m) where
  pure := Pure.pure
  seqLeft := SeqLeft.seqLeft
  seqRight := SeqRight.seqRight

instance [Monad m]: Monad (WriterT m) where
  pure := Pure.pure
  bind := Bind.bind
  map  := Functor.map

def logWriterT [Monad m] (s: String): WriterT.{u} m PUnit.{u+1} :=
  pure (f := m) (.unit, s)
