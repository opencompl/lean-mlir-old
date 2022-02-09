{-# LANGUAGE GADTs #-}
{-# LANGUAGE TypeFamilies #-}
import Data.Map


-- http://casperbp.net/posts/2019-04-nondeterminism-using-a-free-monad/index.html
{-
data FreeCommand c r a where
  FRet :: a -> Free c r a
  -- | R is the response.
  FBind :: c -> (r -> FreeCommand c r a) -> FreeCommand c r a
-}

-- data Free f a = Roll f (Free f a) | Pure a
-- data Cofree f a = Branch (a, f (Cofree f a))


data Rewrite a where
  Focus ::  String -> Rewrite a -> Rewrite a
  Name :: String -> Rewrite a -> Rewrite a
  Op :: String -> Rewrite a -> Rewrite a

-- | state
-- | focus, map from focus to 
newtype S = S (String, Map String (String, [String]))


  
class Group v where
  unit :: v
  gop :: v -> v -> v

class Torsor o v where
  (@-) :: o -> o -> v
  (@+) :: o -> v -> o


  
